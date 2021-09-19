import warnings
import pandas as pd
import numpy as np
import os
#os.environ['R_HOME'] = '/opt/conda/envs/masterenv/lib/R'
# from rpy2 import robjects
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# import tzlocal
from scipy import stats
import matplotlib.pyplot as plt
from itertools import compress, product
from world_viewer.contagion_analysis import ContagionAnalysis



# ISLR = importr('ISLR')
# pandas2ri.activate()

class Glasses(ContagionAnalysis):

    _explained_var = ''
    TEMP_DIR = "tmp/"

    def __init__(self, world, verbose = True):
        self.world = world
        super().__init__(world)


    def load_data(self, explained_var = 'edge', verbose = True, read_cached = False):
        self.data = self.world.get_data(read_cached)
        self.set_explained_var(explained_var, verbose)
        if verbose:
            print('')
            print("Loaded World:")
            print("--------------")
            print(self.data.describe())
            print("")

    def set_explained_var(self, explained_var, verbose):
        # if an explanatory var was already set: reload world
        if self._explained_var != '':
            print('Reload world...')
            self._prep_data(verbose)

        # calc opinion of next time step
        self._explained_var = 'next_' + explained_var
        self.data[self._explained_var] = self.data.groupby(['id_A','id_B'])[explained_var].shift(-1)
        self.data.dropna(inplace=True)
        self.data[self._explained_var] = self.data[self._explained_var].astype('bool')

    def add_unlike_minded_friends_number(self,opinion):
        ulm_friends = self.world.get_unlike_minded_friends_number(opinion)
        data = self.data.reset_index()
        data.set_index(['id_A','time'], inplace=True)
        data = data.join(ulm_friends)
        data['ulm_friends_' + opinion].fillna(0,inplace=True)
        data.reset_index(inplace=True)
        data.set_index(['id_A','id_B','time'],inplace=True)
        self.data = data

    def add_unlike_minded_friends_fraction(self,opinion):
        ulm_friends = self.world.get_unlike_minded_friends_fraction(opinion)
        ulm_friends = ulm_friends.reset_index().rename(columns={'node_id':'id_A'})
        ulm_friends.set_index(['id_A','time'], inplace=True)
        data = self.data.reset_index()
        data.set_index(['id_A','time'], inplace=True)
        data = data.join(ulm_friends)
        data['ulm_friends_' + opinion].fillna(0,inplace=True)
        data.reset_index(inplace=True)
        data.set_index(['id_A','id_B','time'],inplace=True)
        self.data = data

    def run_binned_contagious_analysis(self, n_bins = 5, plot=True):
        exposure_index = "exposure"
        bin_width = self.data[exposure_index].max() / n_bins
        bins = pd.DataFrame()

        # build bins for different exposures
        for i in np.arange(n_bins):
            if i < n_bins-1:
                data_bin = self.data[(self.data[exposure_index] < (i+1) * bin_width) & (self.data[exposure_index] >= i * bin_width)]
            else:
                data_bin = self.data[self.data[exposure_index] >= i * bin_width]

            # calc probabilities to change opinion of nodes in bin
            n = data_bin.node_id.count()
            prob = data_bin[data_bin.op_change == True].node_id.count() / n
            error = np.sqrt(prob * (1-prob) / n )

            # set exposure to center of the bin
            exposure = round((i * bin_width + bin_width/2)*1000 )/1000

            # build df and append
            df = pd.DataFrame([[exposure, prob, error]], columns=['exposure','probability','error'])
            if bins.empty:
                bins = df
            else:
                bins = bins.append(df)

        # plot results
        if plot:
            bins.set_index('exposure',inplace=True)
            fig, ax = plt.subplots(1,1)
            bins.plot(ax=ax, yerr='error', xerr=bin_width/2 , marker='x', linestyle='None',capsize=3, legend=False, lw=1, ms=8)
            plt.ylabel('probability to change opinion')
            plt.ylim(-0.05,1)
            plt.show()

        return bins

    def run_logistic_reg_contagious_analysis(self):
        exposure_index = "exposure"
        self._explained_var = 'op_change'
        data = self.data[[exposure_index,'op_change']].copy()
        data['log_exposure'] = np.log(data[exposure_index]+0.0001)
        [fitted_params, probabilities] = self.run_logistic_regresssion(data = data, verbose = True, plot=False, drop_cols=False, formular="op_change ~ log_exposure+ exposure")
        probabilities.set_index(exposure_index, inplace=True)
        probabilities[['prob','err']].plot(yerr = 'err', linestyle='None',marker='x',capsize=3, legend=False, lw=1, ms=8)
        plt.ylim(-0.05,1)
        plt.ylabel('probability to change opinion')
        plt.show()
        return [fitted_params, probabilities]

    def run_correlation_contagious_analysis(self):


        # calc corrlation for each time step
        corr = self._mean_no_diag(A*D) - self._mean_no_diag(D)*self._mean_no_diag(A)

        #plot correlation over time
        xticks = np.arange(len(time)).astype('str')
        fig, ax = plt.subplots(1,1)
        ax.plot(xticks,corr, marker='x',linestyle='None')
        if opinion != 'op_synthetic':
            plt.xticks(xticks,time,rotation='vertical')
        plt.xlabel('time')
        plt.ylabel('$<d_{ij}\cdot a_{ij}> - <d_{ij}>\cdot <a_{ij}>$')
        plt.tight_layout()
        plt.show()


    def run_logistic_regresssion(self, data = pd.DataFrame(), verbose=True, plot=True, drop_cols = True, formular = None):
        if data.empty: data = self.data.copy()
        if drop_cols:
            # drop node id's and time
            data = data.reset_index()
            data = data.drop(columns=['id_A','id_B','time'])

        n = len(data.columns.values)-1
        if not formular:
            if n > 1:
                formular = self._explained_var + ' ~ .^'+str(n)
            else:
                formular = self._explained_var + ' ~ .'

        print(formular)
        print(data.head())

        # fit in R language
        glm = robjects.r['glm']
        full_model = glm(formular, data = data  , family = 'binomial')

        # get fitted parameters
        summary = robjects.r['summary']
        colnames = robjects.r['colnames']
        rownames = robjects.r['rownames']
        as_v = robjects.r['as.vector']
        sum = summary(full_model)[11]
        fitted_params = pd.DataFrame(np.array([as_v(sum.rx(True, 1)),\
                                               as_v(sum.rx(True, 2)),\
                                               as_v(sum.rx(True, 3)),\
                                               as_v(sum.rx(True, 4))
                                              ]).transpose(),\
                                              columns=colnames(sum), \
                                              index=rownames(sum) )
        if verbose:
            print("Fitted Parameters:")
            print("-------------------")
            print(fitted_params)
            print("")

        # calc predicted probabilities
        predict = robjects.r['predict']
        unique_rows = data.drop(columns=self._explained_var).drop_duplicates()
        prediction = predict(full_model, unique_rows, 'response', True)
        unique_rows['prob'] = prediction[0]
        unique_rows['err'] = prediction[1]
        probabilities = unique_rows.sort_values('prob')
        if verbose:
            print("Estimated Probabilities:")
            print("-------------------------")
            print(probabilities)

        if plot:
            self._bar_plot(probabilities)
        return [fitted_params, probabilities]

    def run_stepwise_logistic_reg(self):
        return 0


    def run_hypothesis_tests(self,verbose = True):

        # drop node id's and time
        data = self.data.reset_index()
        data = data.drop(columns=['id_A','id_B','time'])

        cols = list(data.columns)

        # count frequency of rows
        data['n'] = 0 # frequency of equal rows
        sample = data.groupby(cols).count().reset_index()

        expl_var_next_t = self._explained_var
        expl_var_t = self._explained_var.replace('next_','')
        #print(expl_var_next_t, expl_var_t)

        for explained_var_status in [False, True]:
            n = sample[(sample[expl_var_t] == explained_var_status) & \
                       (sample[expl_var_next_t] != explained_var_status)] \
                    .drop(columns=[expl_var_t, expl_var_next_t])
            explanatory_vars = list(n.columns)
            explanatory_vars.remove('n')
            n.set_index(explanatory_vars,inplace=True)

            n_group = sample[(sample[expl_var_t] == explained_var_status) & \
                             (sample[expl_var_next_t] == explained_var_status)] \
                    .drop(columns=[expl_var_t, expl_var_next_t])
            n_group.set_index(explanatory_vars,inplace=True)
            n_group += n
            # determine reference values where there people are not like-minded
            n_ref = n.loc[tuple([False]*len(explanatory_vars))].values[0]
            n_group_ref = n_group.loc[tuple([False]*len(explanatory_vars))].values[0]
            p_ref = n_ref/n_group_ref

            p_i = n/n_group
            p = (n + n_ref) / (n_group + n_group_ref)
            sigma = np.sqrt(p*(1-p) * (1/n_group + 1/n_group_ref) )
            z_score = (p_i - p_ref)/sigma
            p_value = stats.norm.sf(abs(z_score))*2 # two sided

            # create results dataframe
            result = z_score.rename(columns={'n':'z_score'})
            result['p_value'] = p_value
            result['signif'] = p_value < 0.05
            result = result[result.z_score != 0]
            result.signif.replace(True,'< 0.05',inplace=True)
            result.signif.replace(False,'',inplace=True)

            if verbose:
                print("Hypothesis Tests:")
                print("-------------------------")
                print("Null-Hypothesis: To change " + expl_var_t + " from " + str(explained_var_status) + " to " +  str(not explained_var_status) + " does not depend on " + str(",".join(explanatory_vars))  + ".")
                print(result)
                print("")
        return result

    def _mean_no_diag(self, M):
        n = len(M[1,:,:])
        return np.sum(M,(1,2))/(n*n-n)

    def run_correlation_analysis(self, opinion, normalize = True, read_cached = False):
        # get data
        data = self.world.get_data(read_cached)
        data.reset_index(inplace=True)
        time = data.time.astype('str').unique() #important for later plotting
        time[1::2] = ''

        # fill in for all i-j also j-i
        data = data.append(data.rename(columns={'id_A':'id_B','id_B':'id_A'}),sort=True)
        # convert to matrix, fill all diagonal elements (which are nan) with 0 and cast to int
        data = data.set_index(['time','id_A','id_B']).unstack().fillna(0).astype('int')

        # convert into numpy matrix
        D = np.array(list(data[opinion].groupby('time').apply(lambda p: p.values)))
        A = np.array(list(data.edge.groupby('time').apply(lambda p: p.values)))

        # calc corrlation for each time step
        corr = self._mean_no_diag(A*D) - self._mean_no_diag(D)*self._mean_no_diag(A)

        if normalize:
            sigma_A = np.sqrt( (self._mean_no_diag(A*A) - self._mean_no_diag(A)*self._mean_no_diag(A)))
            sigma_B = np.sqrt( (self._mean_no_diag(D*D) - self._mean_no_diag(D)*self._mean_no_diag(D)) )
            corr /= sigma_A*sigma_B

        #plot correlation over time
        xticks = np.arange(len(time)).astype('str')
        fig, ax = plt.subplots(1,1)
        ax.plot(xticks,corr, marker='x',linestyle='None')
        if opinion != 'op_synthetic':
            plt.xticks(xticks,time,rotation='vertical')
        plt.xlabel('time')
        plt.ylabel('$<d_{ij}\cdot a_{ij}> - <d_{ij}>\cdot <a_{ij}>$')
        plt.tight_layout()
        plt.show()

    def _bar_plot(self, probabilities, title='', print_lables=True):

        probabilities.edge.replace(True, 'bond + like-minded \nin', inplace=True)
        probabilities.edge.replace(False, 'not bond + like-minded \nin', inplace=True)
        probabilities['names'] = probabilities.edge.astype('str')

        for op in probabilities.columns[probabilities.columns.str.startswith('op_')]:
            op_string = op.replace('op_','')
            probabilities[op].replace(True, ' ' + op_string + ',', inplace=True)
            probabilities[op].replace(False, '', inplace=True)
            probabilities['names'] = probabilities['names'] + probabilities[op].astype('str')

        probabilities.names.replace('bond + like-minded \nin', 'bond + not like-minded', inplace=True)
        probabilities.names.replace('not bond + like-minded \nin', 'not bond + \nnot like-minded', inplace=True)
        probabilities.names = probabilities.names.apply(lambda x: x.strip(','))

        probabilities['inv_prob'] = 1-probabilities.prob
        probabilities.prob = round(probabilities.prob *10000)/100.0
        probabilities.inv_prob = round(probabilities.inv_prob *10000)/100.0
        probabilities.err = round(probabilities.err *10000)/100.0


        # plot the bars
        r = np.arange(0,len(probabilities.names))
        barWidth = 0.85
        fig, ax = plt.subplots(1,1)

        ax.bar(r, probabilities.prob, color='#f9bc86', width=barWidth,  label=self._explained_var.replace('next_','') + " = True in next time step", yerr = probabilities.err)
        ax.bar(r, probabilities.inv_prob, bottom=probabilities.prob , color='#a3acff', width=barWidth, label=self._explained_var.replace('next_','') + " = False in next time step")
        plt.xticks(r, probabilities.names, rotation='vertical', fontsize=12)
        plt.xlabel("groups of people",fontsize=12)
        plt.ylabel('transition probability',fontsize=12)
        lgd = plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
        plt.title(title)

        # write percentages as labels on top of the bars
        x_offset = -0.3
        y_offset = -6
        if print_lables:
            for p in ax.patches:
                b = p.get_bbox()
                if not ( (p.get_height() < 5) & (b.y1 > 95) ):
                    ax.annotate(str(p.get_height()) + "%", ((b.x0 + b.x1)/2 + x_offset, abs(b.y1 + y_offset)))

        plt.tight_layout()
        plt.show()
        # write figure to disc
        fig.savefig("tmp/" + title.replace(' ', '_') + ".png", dpi=400, bbox_extra_artists=(lgd,), bbox_inches='tight')
