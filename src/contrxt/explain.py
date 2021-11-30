import os
from typing import List, Dict
from time import time
from collections import defaultdict
import logging
import stopit
import pandas as pd
from graphviz import Source
from pyeda.inter import expr, expr2bdd

from contrxt.util.helpers import union, jaccard_distance
from contrxt.util.logger import build_logger
from contrxt.util.BDD2Text import BDD2Text

class Explain():
    '''
    '''

    def __init__(self, data_manager,
                 save_path, graphviz_path,
                 log_level=logging.INFO,
                 save_bdds=True, save_csvs=True):
        '''Initialize explain object
        '''

        self.logger = build_logger(log_level, __name__, 'logs/explain.log')
        os.environ['PATH'] += os.pathsep + graphviz_path

        self.data_manager = data_manager
        self.save_path = save_path
        self.bdd_img_path = f'{save_path}/add_del'
        self.filter = filter
        self.bdd_dict = self.load_bdd_data()

        self.save_bdds = save_bdds
        self.save_csvs = save_csvs
        self.results = defaultdict(lambda: {})

        if os.path.exists(f'{self.save_path}/paths_add_del.csv'):
            os.remove(f'{self.save_path}/paths_add_del.csv')

    def load_bdd_data(self):
        '''Load bdd data
        '''
        bdd_dict = defaultdict(lambda: list())
        trace_results_df = pd.read_csv(f'{self.save_path}/trace.csv', decimal=',', sep=';')
        for _, row in trace_results_df.iterrows():
            class_id = str(row['class_id'])
            bdd_string = row['bdd_string']
            bdd_dict[class_id].append(bdd_string)
        return bdd_dict

    def _obdd_diff(self, class_id, s1, s2):
        '''Calculates differences and KPIs between two BDDs
        '''
        start_time = time()

        f = expr2bdd(expr(s1))
        g = expr2bdd(expr(s2))

        f_add_bdd = ~f & g
        f_del_bdd = f & ~g
        f_still_bdd = f & g

        sat_add_lst = list(f_add_bdd.satisfy_all())
        sat_del_lst = list(f_del_bdd.satisfy_all())
        sat_still_lst = list(f_still_bdd.satisfy_all())

        set_features_f = set(f.inputs)
        set_features_g = set(g.inputs)
        set_features_add = set_features_f - set_features_g
        set_features_del = set_features_g - set_features_f
        set_features_still = set_features_f.union(set_features_g) - set_features_f.intersection(set_features_g)
        self.logger.debug(f'SET FEATURES F: {set_features_f}')
        self.logger.debug(f'SET FEATURES G: {set_features_g}')
        self.logger.debug(f'SET FEATURES REMOVE ADD: {set_features_add}')
        self.logger.debug(f'SET FEATURES REMOVE DEL: {set_features_del}')
        self.logger.debug(f'SET FEATURES REMOVE STILL: {set_features_still}')
        self.logger.debug('-----------------------------------')
        self.logger.debug(f'SAT ADD PATHS BEFORE REMOVING DIFFS: {sat_add_lst}')
        self.logger.debug(f'SAT DEL PATHS BEFORE REMOVING DIFFS: {sat_del_lst}')
        self.logger.debug(f'SAT STILL PATHS BEFORE REMOVING DIFFS: {sat_still_lst}')
        self.logger.debug('-----------------------------------')

        sat_add_lst = self._remove_diffs(sat_add_lst, set_features_add)
        sat_del_lst = self._remove_diffs(sat_del_lst, set_features_del)
        sat_still_lst = [str({str(key): value for key, value in x.items()}) for x in sat_still_lst if len(set(x.keys()).intersection(set_features_still))==0]

        sat_add_lst = {str(x) for x in sat_add_lst}
        sat_del_lst = {str(x) for x in sat_del_lst}

        self.logger.debug(f'SAT ADD PATHS AFTER REMOVING DIFFS: {sat_add_lst}')
        self.logger.debug(f'SAT DEL PATHS AFTER REMOVING DIFFS: {sat_del_lst}')
        self.logger.debug(f'SAT STILL PATHS AFTER REMOVING DIFFS: {sat_still_lst}')
        self.logger.debug('-----------------------------------')

        # Reorganize bdds after removing features. For still no removal needed
        f_add_bdd, sat_add_lst = self.dict2bdd_paths(sat_add_lst)
        f_del_bdd, sat_del_lst = self.dict2bdd_paths(sat_del_lst)
        f_still_bdd, sat_still_lst = self.dict2bdd_paths(sat_still_lst)

        sat_add = len(sat_add_lst)
        sat_del = len(sat_del_lst)
        sat_still = len(sat_still_lst)
        self.results[class_id]['sat_add'] = sat_add
        self.results[class_id]['sat_del'] = sat_del
        self.results[class_id]['sat_still'] = sat_still

        self.logger.debug(f'LENGTHS OF SAT_ADD, SAT_DEL, SAT_STILL: {sat_add} {sat_del} {sat_still}')

        if self.save_csvs:
            self._save_kpi_csv(class_id, sat_add_lst, 'add')
            self._save_kpi_csv(class_id, sat_del_lst, 'del')
            self._save_kpi_csv(class_id, sat_still_lst, 'still')

        if self.save_bdds:
            self._save_bdd(f'{class_id}_add', f_add_bdd)
            self._save_bdd(f'{class_id}_del', f_del_bdd)
            self._save_bdd(f'{class_id}_still', f_still_bdd)

        # Calculate KPIs
        self.results[class_id]['add'] = self.calculate_kpi(sat_add, sat_del, sat_still)
        self.results[class_id]['del'] = self.calculate_kpi(sat_del, sat_add, sat_still)
        self.results[class_id]['still'] = self.calculate_kpi(sat_still, sat_add, sat_del)
        self.results[class_id]['j'] = jaccard_distance(set_features_f, set_features_g)

        self.results[class_id]['s1'] = str(len(set_features_f))
        self.results[class_id]['s2'] = str(len(set_features_g))
        self.results[class_id]['union'] = str(union(set_features_f, set_features_g))
        self.results[class_id]['runtime'] = round(time() - start_time, 3)

        self.logger.info(self.results[class_id])
        return True

    @staticmethod
    def _remove_diffs(lst, set_features):
        '''Remove differences between list and set of features
        '''
        lst = [{str(key): value for (key, value) in x.items()} for x in lst]
        set_features = {str(x) for x in set_features}
        for i, path in enumerate(lst):
            to_pop = []
            for key in path.keys():
                if key in set_features:
                    to_pop.append(key)
            for key in to_pop:
                del lst[i][key]
        return lst

    @staticmethod
    def dict2bdd_paths(dct: Dict) -> List:
        """Transforms a dictionary to a list of BDD paths.

        Parameters
        ----------
        dct : Dict
            Dictionary of features.

        Returns
        -------
        out: List
            List of paths that satisfy the generated BDD.

        """

        if len(dct) == 0:
            return expr2bdd(expr(None)), []
        path_list = []
        for path in dct:
            temp_path = path.replace('{', '').replace('}', '').replace("'", '')
            temp_path = {i.split(': ')[0]: i.split(': ')[1] for i in temp_path.split(', ')}
            criteria_list = []
            for criteria in temp_path.items():
                if criteria[1] == '0':
                    criteria_list.append(f'~{criteria[0]}')
                else:
                    criteria_list.append(f'{criteria[0]}')
            path_list.append(' & '.join(criteria_list))
        bdd_str = ' | '.join(path_list)
        funct = expr(bdd_str)
        bdd = expr2bdd(funct)
        sat_lst = list(bdd.satisfy_all())
        sat_lst = {str(x) for x in sat_lst}
        return bdd, sat_lst

    @staticmethod
    def calculate_kpi(main_sat, other_sat, last_sat):
        '''Calculate KPI (ADD/DEL)
        '''
        try:
            result = (main_sat) / (main_sat + other_sat + last_sat)
        except ZeroDivisionError:
            result = 0
        return result

    @staticmethod
    def calculate_kpi_global(df):
        '''Calculate KPI Global
        '''
        max_sat_add = df['sat_add'].max()
        max_sat_del = df['sat_del'].max()
        max_sat_still = df['sat_still'].max()
        df['add_global'] = [x if x==x else 0 for x in df['sat_add'] / max_sat_add]
        df['del_global'] = [x if x==x else 0 for x in df['sat_del'] / max_sat_del]
        df['still_global'] = [x if x==x else 0 for x in df['sat_still'] / max_sat_still]
        return df

    def calculate_malandri(self):
        return 0

    def start_comparison(self, class_id):
        '''Begins comparison of one class
        '''
        with stopit.ThreadingTimeout(72000) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            self.logger.info(f'Starting computation for class {class_id}')

            try:
                d_1 = self.bdd_dict[class_id][0]
                d_2 = self.bdd_dict[class_id][1]
            except KeyError:
                self.logger.exception('Missing BDD')
                return False

            self._obdd_diff(class_id, d_1, d_2)
            self.logger.info(f'Finishing class {class_id}, time: {self.results[class_id]["runtime"]}')

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            self.logger.info(f'Exiting class {class_id}')
            return True
        self.logger.warning(f'Timeout for class {class_id}')
        return False

    def BDD2Text_single(self, class_id):
        '''Prints BDD2Text for class
        '''
        text = BDD2Text(f'{self.save_path}/paths_add_del.csv', class_id, 85)
        text.simple_text()

    def BDD2Text(self):
        '''Prints BDD2Text for all classes
        '''
        for class_id in self.bdd_dict:
            text = BDD2Text(f'{self.save_path}/paths_add_del.csv', class_id, 85)
            text.simple_text()

    def run_explain(self):
        '''Run explain for each class
        '''
        for class_id in self.bdd_dict:
            self.start_comparison(class_id)

        if self.save_csvs:
            self._save_results()

    def _save_results(self):
        '''Save the results to csv
        '''
        path = f'{self.save_path}/explain.csv'
        self.logger.info(f'Saving results to {path}')
        kpi_df = pd.DataFrame.from_dict(self.results, orient='index').reset_index()
        kpi_df = kpi_df.rename(columns={'index': 'class_id'})
        kpi_df = self.calculate_kpi_global(kpi_df)
        kpi_df = kpi_df[['class_id', 'add', 'del', 'still',
                         'add_global', 'del_global', 'still_global',
                         'sat_add', 'sat_del', 'sat_still', 'j',
                         's1', 's2', 'union', 'runtime']]
        kpi_df = kpi_df.round(3)
        try:
            kpi_df.to_csv(path, index=False, decimal='.', sep=';')
        except PermissionError:
            self.logger.error('Error: cannot save results, file in use!')

    def _save_kpi_csv(self, class_id, bdd, type_):
        '''Save the csv with paths for each kpi
        '''
        for rule in bdd:
            count_t1 = self.data_manager['time_1'].count_rule_occurrence(rule)
            count_t2 = self.data_manager['time_2'].count_rule_occurrence(rule)
            count_tot = str(count_t1 + count_t2)
            csv_row = ';'.join([class_id, rule, type_, count_tot])
            with open(f'{self.save_path}/paths_add_del.csv', 'a', encoding='utf-8') as file:
                file.write(csv_row)
                file.write('\n')

    def _save_bdd(self, name, bdd):
        '''Save the BDD to pdf file. Remove useless non pdf file
        '''
        self.logger.info(f'Printing {name} pdf file')
        graph_viz = Source(bdd.to_dot())
        path = f'{self.bdd_img_path}/{name}'
        graph_viz.render(path, view=False)
        if os.path.exists(path):
            os.remove(path)

    def _save_bdd2text(self):
        '''
        '''
        pass
