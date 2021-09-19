import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import itertools
import datetime
import os
from math import sqrt
#import seaborn as sns

class ContagionAnalysis():

    def __init__(self, world):
        self.world = world
        # time as lable to write files
        now = datetime.datetime.now()
        self.now = now.strftime("%Y-%m-%d_%H:%M")

    def run_contaigon_analysis(self, opinion_type, analysis="expo_frac", n_bins = 20, binning = True, save_plots = False, show_plot=True, write_data = True, output_folder = ""):
        ''' Makes a full contagion analysis  
            Parameters:
                opinion_type: (str) name of trait
                analysis: (str) name of analysis type (expo_frac, expo_nmb)
                n_bins: (int) number of bins
                binning: (bool) if to do binning
                save_plots: (bool) if to save plot on hd
                write_data: (bool) if to write data on hd
                ouput_folder: (str) folder to save data + plots
        '''

        # name to lable files
        name =  self.world.name + \
                "_" + analysis + \
                "_" + self.now
        self.output_folder = output_folder
        print("Write into: " + self.TEMP_DIR + output_folder)
        if not os.path.exists(self.TEMP_DIR + output_folder):
            os.makedirs(self.TEMP_DIR + output_folder)

        # calc exposure
        exposure = self.calc_exposure(analysis, opinion_type)
        #write data
        if write_data:
            exposure.to_pickle(self.TEMP_DIR + output_folder + "exposure_" + name + ".pkl")

        # calc trait change
        data, expo_agg = self.opinion_change_per_exposure(exposure, opinion_type)
        #write data
        if write_data:
            data.to_pickle(self.TEMP_DIR + output_folder + "data_" + name + ".pkl")

        # plot
        plot_data = self.plot_opinion_change_per_exposure_number(data, analysis, binning, n_bins, \
                    save_plots, show_plot)

        return [data, plot_data]

    def _get_neighbors(self,g,i):
        ''' returns neighbors of node i in graph g '''
        try:
            return [n for n in g[i]]
        except KeyError:
            return []

    def _calc_expo_frac(self, node_id, opinion_type, t, op_nodes, graph, all_opinions):
        ''' Calculate exposure as fraction of encounters to people with other opinion '''
        neighbors = self._get_neighbors(graph, node_id)
        opinions = op_nodes.loc[neighbors]
        
        nmb_1 = opinions.loc[opinions[opinion_type] == True, opinion_type].count()
        nmb_2 = opinions.loc[opinions[opinion_type] == False, opinion_type].count()      

        exposure = pd.DataFrame({ opinion_type: [True, False],\
                                  'n_influencer': [nmb_1, nmb_2],\
                                  'frac_influencer': [nmb_1, nmb_2] })
        
        if (len(neighbors) <= 2) & (self.world.type == "SYN"):
            if self.world.cc == True:
                exposure *= 0

        # normalize exposure
        if len(neighbors) > 0:
            exposure.frac_influencer /= len(neighbors)

        exposure['n_nbs'] = len(neighbors)
        exposure['node_id'] = node_id
        exposure['time'] = t
        return exposure

    def calc_exposure(self, analysis, opinion_type, exposure_time = 7):
        ''' Calculate exposure for opinion type, distinguish between different analysis types '''
        print("INFO: Calc exposure...")

        # prepare some varibales for late use
        all_opinions = pd.DataFrame( self.world.op_nodes[opinion_type].unique(), \
                                     columns=[opinion_type])
        nodes = self.world.op_nodes.node_id.unique()
        self.world.op_nodes.time = pd.to_datetime(self.world.op_nodes.time)
        op_nodes = [self.world.op_nodes[self.world.op_nodes.time == t].set_index('node_id') \
                        for t in self.world.time.time]

        # distinguish between analysis types and calc exposure
        if analysis == "expo_frac":
            print("INFO: Calc expo frac")
            expo = []
            for t in self.world.time.time:
                rel_graph = self.world.get_relation_graph_t(t = t)
                op_nodes_t = self.world.op_nodes.loc[self.world.op_nodes.time == t].set_index('node_id')
                expo += [ self._calc_expo_frac( node_id, opinion_type, t, op_nodes_t, rel_graph, all_opinions) \
                         for node_id in nodes]

            expo = pd.concat(expo)

        # calc mean over last exposure_time days
        sigma = pd.to_timedelta(exposure_time, unit='d').total_seconds() #seconds
        two_sigma_sqr = 2* sigma * sigma

        expo.time = pd.to_datetime(expo.time)
        expo = expo.groupby(['node_id',opinion_type])["time", "n_influencer", "n_nbs", "frac_influencer"].apply( \
                                                        lambda p: self._agg_expo(p, two_sigma_sqr, analysis) \
                                                     ).reset_index()
        if analysis == "expo_frac":
            expo.set_index(['node_id','time',opinion_type],inplace=True)
            expo["exposure"] = expo.n_influencer_mean / expo.n_nbs_mean
            expo.reset_index(inplace=True)

        expo.set_index(['node_id','time'],inplace=True)
        return expo

    def _agg_expo(self, expo_slice, two_sigma_sqr, analysis):
        ''' weighted temporal mean of expo_slice  '''
        expo_slice = expo_slice.copy()
        expo_slice.time = expo_slice.time.astype('int')/1000000000.0 # to seconds
        time_matrix = np.array([expo_slice.time.values]*len(expo_slice.time))
        diff = (time_matrix - time_matrix.transpose()) #seconds
        matrix = np.exp(-(diff * diff)/two_sigma_sqr)
        filter_past = np.tril(np.ones_like(matrix))
        matrix *= filter_past
        if analysis == "expo_nmb":
            expo_slice["exposure"] =  np.dot(matrix, expo_slice.exposure)
        else:
            norm = np.dot(matrix, np.ones_like(expo_slice.frac_influencer))
            expo_slice["frac_influencer_mean"] =  np.dot(matrix, expo_slice.frac_influencer)
            expo_slice["frac_influencer_mean"] /= norm
            expo_slice["n_influencer_summed"] = np.dot(matrix, expo_slice.n_influencer)
            expo_slice["n_influencer_mean"] = expo_slice["n_influencer_summed"] / norm
            expo_slice["n_nbs_summed"] =  np.dot(matrix, expo_slice.n_nbs) 
            expo_slice["n_nbs_mean"] = expo_slice["n_nbs_summed"] / norm
        expo_slice.time = pd.to_datetime(expo_slice.time, unit="s")
        return expo_slice.set_index("time")

    def opinion_change_per_exposure(self, exposure, opinion_type,  opinion_change_time = 1):
        ''' calculated if the opinion changed '''
        print("INFO: Calc op-change")
        exposure = exposure.copy()
        op_nodes = self.world.op_nodes.copy()
        op_nodes.time = pd.to_datetime(op_nodes.time)
        op_nodes.set_index(['node_id','time'],inplace=True)

        exposure['curr_opinion'] = op_nodes[opinion_type]
        exposure.reset_index(inplace=True)

        # calc prev opinion
        exposure.sort_values('time',inplace=True)
        exposure['prev_opinion'] = exposure.groupby(['node_id', opinion_type])['curr_opinion'].shift(opinion_change_time)
        exposure['prev_time'] = exposure.groupby(['node_id', opinion_type])['time'].shift(opinion_change_time)
        exposure.dropna(subset=["prev_opinion"] ,inplace=True)

        #calc op change
        exposure['op_change'] = exposure.curr_opinion == exposure[opinion_type]

        #kick out exposure to own opinions
        exposure.sort_index(inplace=True)
        exposure.loc[(exposure.prev_opinion == exposure[opinion_type]),'op_change'] = np.nan
        exposure.dropna(subset=["exposure","op_change"] ,inplace=True)

        data = exposure[['exposure','op_change', opinion_type]]

        return data, exposure
        
        

    def plot_opinion_change_per_exposure_number(self, data, analysis,
            binning, n_bins=5, save_plots = False, show_plot = False, limits = True, suffix = "", ax = None, fig = None, ci = False, label="", y_lower_lim = -0.01, y_upper_lim = 0.2, q_binning = False, plt_diag = False, loglog = False, log_binning= False, step_plot = True, color = "k", min_bin_size=30, max_bin_size=None, lable_outer=True, legend_loc="upper left", bin_width=1, x_lim=None, legend=True, xlabel = None, marker = "None", retbins=False, bins=None, ylabel="", markersize=9, borders=False, grid=True, retdata=False, plot_only = False):
        ''' calculate dose response function and plots it 
            Parameters:
                data: input data generated by opinion_change_per_exposure()
                analysis: method used: expo_frac or expo_nmb
                binning: bool if to use a binning method
                n_bins: number of bins
                save_plots: (bool) write plots on drive
                show_plots: (bool) print plots
                limits: (bool) create xlims
                suffix: (string) suffix for filename
                ax: ax to attache figure
                fig: fig to attache figure
                ci: (bool) plot 95 confidence intervall
                label: (str) label for the data in the legend
                y_lower_lim: (float)
                y_upper_lim: (float)
                q_binning: (bool) if make equal quantile binning
                plt_diag: (bool) if plot a diagonal f(x) = x
                loglog: (bool) if double log plot
                log_binning: (bool) if make equal bin size in log space
                step_plot: (bool) if plotting bins as horizontal lines
                color: (string) color of plot
                min_bin_size: (int) minimal number of data points in bin. If less, then bin will be dropped.
                lable_outer: dropp inner lables if there are multiple plots
                legend_loc: location of legend
        '''

        data = data.copy()
        if not plot_only:
            #calc bins
            if binning:
                if not bins == None:
                    cut = pd.cut(data['exposure'].values,bins=bins)
                    data.exposure = [b.mid for b in cut]
                    if step_plot:
                        data["exposure_min"] = [b.left for b in cut]
                    data = data[data.exposure > 0] # the cut function produces one bin with exposure -0.05
                elif q_binning: # equal quantile binning
                    if retbins:
                        cut, bins = pd.qcut(data['exposure'].values, n_bins, duplicates="drop", retbins=True)
                    else:
                        cut = pd.qcut(data['exposure'].values, n_bins, duplicates="drop")
                    data.exposure = [b.mid for b in cut]
                    if step_plot:
                        data["exposure_min"] = [b.left for b in cut]
                    data = data[data.exposure > 0] # the qcut function produces one bin with exposure -0.05
                elif log_binning: # binning with equal bin size on log scale
                    bins = pd.cut(data.exposure.values, np.logspace(-3,0,n_bins))
                    expo_tmp = []
                    expo_tmp_min = []
                    for b in bins:
                        try:
                            expo_tmp += [b.mid]
                            expo_tmp_min += [b.left]
                        except AttributeError:
                            expo_tmp += [b]
                            expo_tmp_min += [b]
                    data.exposure = expo_tmp
                    data["exposure_min"] = expo_tmp_min
                else: # binning with exqual bin size
                    bin_width = self._exposure_to_bins(data, n_bins,analysis=analysis, bin_width= bin_width)
                    data["exposure_min"] = data.exposure - bin_width/2

            # calc dose response function, the error and the 95% confidence intervall
            n = data.groupby('exposure').count()

            if max_bin_size:
                plot_data = data.groupby('exposure').agg(lambda d: d.iloc[:max_bin_size].mean())
                n.loc[n.op_change > max_bin_size] = max_bin_size
            else:             
                plot_data = data.groupby('exposure').mean()
            plot_data['error'] = np.sqrt(plot_data.op_change * (1 + plot_data.op_change) / n.op_change)
            plot_data["confidence"] = 1.96*plot_data['error']  #data.groupby("exposure").op_change.agg(lambda d: self._confidence(d.sum(), d.count()))
            plot_data["n"] = n.op_change
            plot_data = plot_data[plot_data.n > min_bin_size]
        
        if plot_only:
            plot_data = data
        
        #plot_data['error'] = np.sqrt(plot_data.op_change * (1 - plot_data.op_change) / n.op_change + np.cov(plot_data.op_change.values)/(n.op_change * n.op_change)  )
        #plot_data['error'] = np.sqrt(plot_data.op_change * (1 - plot_data.op_change) / n.op_change \
        #                             + (n.op_change -1)/n.op_change * (1/4 - plot_data.op_change*plot_data.op_change) )
        plot_data.dropna(inplace=True)
        #plot_data = plot_data[['op_change','error',"confidence"]]

        # prepare figure
        if not ax:
            fig, ax = plt.subplots(1,1)

        # plot
        #sns.set()
        if binning:
            plot_data.reset_index(inplace=True)
            
            if step_plot:
                line = ax.errorbar(plot_data.exposure, plot_data.op_change, yerr=plot_data.error, \
                             xerr=(plot_data.exposure - plot_data.exposure_min), linestyle='None', \
                             lw=1, color=color, label=label, marker=marker, markersize=markersize)
                if not borders:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                #ax.spines["bottom"].set_visible(False)
                ax.grid(grid, linestyle='--', alpha=0.3)

            else:
                plt.errorbar(plot_data.exposure, plot_data.op_change, yerr=plot_data.error, xerr=(plot_data.exposure - plot_data.exposure_min),\
                        linestyle='None', lw=1, ms=8, label=label, color=color)

        else:
            plot_data.plot(ax=ax, yerr = 'error', marker = 'x',linestyle='None',capsize=3, legend=False, lw=1, ms=8)

        if ci:
            ci_data = plot_data.reset_index()
            left_value = ci_data.loc[ ci_data.exposure == ci_data.exposure.min() ].copy()
            right_value = ci_data.loc[ ci_data.exposure == ci_data.exposure.max() ].copy()
            left_value["exposure"] = left_value["exposure_min"]
            right_value["exposure"] = 2 * right_value["exposure"] - right_value["exposure_min"]
            ci_data = ci_data.append(right_value).append(left_value)
            ci_data.sort_values("exposure",inplace=True)
            ax.fill_between(ci_data.exposure, ci_data.op_change + ci_data.confidence, ci_data.op_change - ci_data.confidence, alpha = 0.4, label = '95% confidence', color=color)

        if loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(0.005,1.1)
            #ax.set_xlim(0.01,1)
            limits = False

        # make plot nice
        
        if not loglog: 
            ax.set_ylim(y_lower_lim,y_upper_lim)
        if analysis == "expo_nmb":
            if xlabel:
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel(r'absolute exposures $K$')
            ax.set_ylabel(ylabel + r"$p_{o\rightarrow o'}(K)$")
            if x_lim: ax.set_xlim(0,x_lim)
        elif analysis == "expo_frac":
            ax.set_xlabel(r"relative exposure $x$")
            ax.set_ylabel(r"$p_{o\rightarrow o'}(x)$")
            if plt_diag: plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), ":k")
            if limits:
                ax.set_xlim(-0.03,1.03)

        if legend: ax.legend(loc = legend_loc)
        if lable_outer: ax.label_outer()

        # show & save plot
        if show_plot: plt.show()
        if save_plots:
            name =  self.world.name + \
                    suffix +\
                    "_BIN" + str(n_bins) + \
                    "_" + analysis + \
                    "_" + self.now
            fig.savefig(self.TEMP_DIR + self.output_folder  + name + ".pdf" \
                    , bbox_inches='tight')
        if retbins:
            return fig, ax, line, bins
        if retdata:
            return fig, ax, line, plot_data
        else:
            return fig, ax, line

    def _exposure_to_bins(self, data, n_bins, analysis, bin_width = 5):
        ''' create qual size bins '''
        exposure_index = "exposure"
        if analysis == "expo_frac":
            bin_width = 1 / n_bins
        elif analysis == "expo_nmb":
            n_bins = data.exposure.max() / bin_width


        # build bins for different exposures
        for i in np.arange(n_bins):
            exposure = round((i * bin_width + bin_width/2)*1000 )/1000
            if i < n_bins-1:
                data.loc[(data[exposure_index] < (i+1) * bin_width) &
                        (data[exposure_index] >= i * bin_width), exposure_index] = exposure
                data.loc[(data[exposure_index] > - (i+1) * bin_width) &
                        (data[exposure_index] <= - i * bin_width), exposure_index] = - exposure
            else:
                data.loc[data[exposure_index] >= i * bin_width, exposure_index] = exposure
                data.loc[data[exposure_index] <= - i * bin_width, exposure_index] = - exposure
        return bin_width

    def _confidence(self, ups, n):
        ''' calculate 95 confidence intervall '''

        z = 1.96
        p = float(ups) / n

        #Wilson score interval
        #https://stackoverflow.com/questions/10029588/python-implementation-of-the-wilson-score-interval
        #left = p + 1/(2*n)*z*z
        #right = z*sqrt(p*(1-p)/n + z*z/(4*n*n))
        #under = 1+1/n*z*z

        #return (left - right) / under

        return z * sqrt(p*(1-p)/n)


