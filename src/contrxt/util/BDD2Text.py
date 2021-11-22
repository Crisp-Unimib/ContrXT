import ast
import re
import textwrap

import pandas as pd
from apyori import apriori

class BDD2Text(object):
    def __init__(self, file_path, cls, wrapp_size=100):
        self.threshold = 0.8
        df = self.prepare_data(file_path)
        self.label = cls
        ######################### add paths ###################################
        self.path_dict_add = df[(df['class'] == cls)
                                & (df['mode'] == 'add')]['path'].tolist()[0]
        N_add = df[(df['class'] == cls)
                                & (df['mode'] == 'add')]['N'].tolist()[0]
        self.N_add = self.rule_trimmer(N_add, self.threshold)
        self.used_paths_add = [x for i, x in enumerate(self.path_dict_add) if self.N_add[i] == True]

        ######################### del paths ###################################
        self.path_dict_del = df[(df['class'] == cls)
                                & (df['mode'] == 'del')]['path'].tolist()[0]
        N_del = df[(df['class'] == cls)
                                & (df['mode'] == 'del')]['N'].tolist()[0]
        self.N_del = self.rule_trimmer(N_del, self.threshold)
        self.used_paths_del = [x for i, x in enumerate(self.path_dict_del) if self.N_del[i] == True]

        ######################### still paths ###################################
        self.path_dict_still = df[(df['class'] == cls)
                                & (df['mode'] == 'still')]['path'].tolist()[0]
        N_still = df[(df['class'] == cls)
                                & (df['mode'] == 'still')]['N'].tolist()[0]
        self.N_still = self.rule_trimmer(N_still, self.threshold)
        self.used_paths_still = [x for i, x in enumerate(self.path_dict_still) if self.N_still[i] == True]
        self.path_length = len(df[(df['class'] == cls) & (df['mode'] == 'still')])

        self.wrapper = textwrap.TextWrapper(width=wrapp_size)

    def text_formatter(self,
                       text:str,
                       bc=None,
                       tc=None,
                       bold:bool=False,
                       underline:bool=False,
                       _reversed:bool=False):
        """Add requested style to the fgiven string.

        Adds ANSI Escape codes to add text color, background color and
        other decorations like bold, undeline and reversed

        Args:
            text: target string
            bc: Background color code (int)
            tc: Text color code (int)
            bold: if True makes the given string bold
            underline: if True makes the given string undelined
            reversed: if True revreses the background and text colors

        """

        assert isinstance(text, str), f'text should be string not {type(text)}'
        assert isinstance(
            bc, (int, type(None))), f'Background color code should be integer not {type(bc)}'
        assert isinstance(
            tc, (int, type(None))), f'Text color code should be integer not {type(tc)}'
        assert isinstance(
            bold, bool), f'Bold should be Boolean not {type(bold)}'
        assert isinstance(
            underline, bool), f'Underline should be Boolean not {type(underline)}'
        assert isinstance(
            _reversed, bool), f'Reversed should be Boolean not {type(_reversed)}'

        if bc is not None:
            bc = f'\u001b[48;5;{bc}m'
        else:
            bc = ''
        if tc is not None:
            tc = f'\u001b[38;5;{tc}m'
        else:
            tc = ''
        if bold:
            b = '\u001b[1m'
        else:
            b = ''
        if underline:
            u = '\u001b[4m'
        else:
            u = ''
        if _reversed:
            r = '\u001b[7m'
        else:
            r = ''

        return(f'{b}{u}{r}{bc}{tc}{text}\u001b[0m')

    def prepare_data(self, file_path):
        temp = pd.read_csv(file_path, sep=';', header=None)
        temp.columns = ['class', 'path', 'bdd', 'N']
        temp = temp[['class', 'bdd', 'path', 'N']]
        temp['class'] = temp['class'].astype('str')

        def string_to_dict(string):
            """stolen from https://stackoverflow.com/a/58561819/5842939"""

            # Add quotes to dict keys
            string = re.sub(r'(\w+):', r'"\1":', string)

            def add_quotes_to_lists(match):
                return re.sub(r'([\s\[])([^\],]+)', r'\1"\2"', match.group(0))

            # Add quotes to list items
            string = re.sub(r'\[[^\]]+', add_quotes_to_lists, string)
            # Evaluate the dictionary
            final = ast.literal_eval(string)
            return final

        def remove_ampamp(dic):
            try:
                del dic['ampamp']
            except KeyError:
                pass
            return dic

        temp['path'] = temp['path'].apply(string_to_dict)
        temp['path'] = temp['path'].apply(remove_ampamp)
        # putthing together paths for each class
        res = []
        for cls in temp['class'].unique():
            for mode in ['add', 'del', 'still']:
                tempak = temp[(temp['class'] == cls) & (temp['bdd'] == mode)]
                paths = tempak['path'].tolist()
                Ns = tempak['N'].tolist()
                res.append([cls, mode, paths, Ns])

        df = pd.DataFrame(res, columns=['class', 'mode', 'path', 'N'])

        return df

    def rule_trimmer(self, data, threshold=0.8):
        """returns a list of T/F for a given list based on their
        cummulative frequency and the given threshold"""

        #print(f'data: {data}')
        summ = sum(data)
        frac = [0 if summ==0 else x/summ for x in data]
        #print(f'frac: {frac}')
        data2 = list(enumerate(frac))
        data2.sort(key=lambda x: x[1], reverse=True)
        #print(f'data2: {data2}')
        curr = 0
        res = []
        for i in data2:
            if curr <= threshold:
                pp = True
                curr += i[1]
            else:
                pp = False
            res.append((i[0], pp))

        res.sort(key=lambda x : x[0])

        return [x[1] for x in res]

    def abs_bads(self, kind):
        """
        returns the token that exists in the all (add/del) paths
        which end as 0 class
        """
        if kind == 'add':
            dicts = self.path_dict_add
        elif kind == 'del':
            dicts = self.path_dict_del
        elif kind == 'still':
            dicts = self.path_dict_still
        res = []
        all_items = set()
        for item in dicts:
            for i in item.keys():
                all_items.add(i)
        for k in all_items:
            if all(k in item and item[k] == 0 for item in dicts):
                res.append(k)
        return res

    def abs_goods(self, kind):
        """
        returns the token that exists in the all (add/del) paths
        which end as 1 class
        """
        if kind == 'add':
            dicts = self.path_dict_add
        elif kind == 'del':
            dicts = self.path_dict_del
        elif kind == 'still':
            dicts = self.path_dict_still
        res = []
        all_items = set()
        for item in dicts:
            for i in item.keys():
                all_items.add(i)
        for k in all_items:
            if all(k in item and item[k] == 1 for item in dicts):
                res.append(k)
        return res

    def remove_key(self, dic, key):
        try:
            del dic[key]
        except KeyError:
            pass

    def dict_to_list(self, d):
        return [x[0] + str(x[1]) for x in d.items()]

    def simple_text(self):
        def agg_0_1(dictak):
            """
            returns the following lists:
                tokens which ended as 0
                tokens which ended as 1
            """
            zeros = [item[0] for item in dictak.items() if item[1] == 0]
            ones = [item[0] for item in dictak.items() if item[1] == 1]
            return zeros, ones

        def list_to_string(data, add_feature=True):
            """
            taking a list of features, returns a string which lists the features
            """
            if len(data) > 1:
                t = ''
                data_1 = data[:-1]
                last = data[-1]
                for i in data_1:
                    i = self.text_formatter(i, bold=True)
                    t += f'{i}, '
                last = self.text_formatter(last, bold=True)
                t += f'and {last}'
                if add_feature:
                    t += ' features'
            elif len(data) == 1:
                t = self.text_formatter(data[0], bold=True)
                if add_feature:
                    t += ' feature'
            else:
                t = 'XXXX'
            return t

        def list_to_string_2(data, text_type):
            """
            taking a list of features, returns a string which lists the COLORED features
            """
            t = data
            if text_type == 'pos':
                tc = 10 # green
            else:
                tc = 9 # red
            if len(data) > 1:
                t = ''
                data_1 = data[:-1]
                last = data[-1]
                for i in data_1:
                    i = self.text_formatter(i, tc=tc, bold=True)
                    t += f'{i}, '
                last = self.text_formatter(last, tc=tc, bold=True)
                t += f'and {last}'

            elif len(data) == 1:
                t = self.text_formatter(data[0], tc=tc, bold=True)


            return t

        def rules_to_shared(data):
            nums = [int(x[-1]) for x in data]
            words = [x[:-1] for x in data]
            pos_words = [x[:-1] for x in data if x[-1] == '1']
            neg_words = [x[:-1] for x in data if x[-1] == '0']
            if len(data) > 1:
                if sum(nums) == 0:
                    return f'the document must {self.text_formatter("not", tc=9, underline=True)} contain {list_to_string_2(words, "neg")}.'
                if sum(nums) == 1:
                    return f'the document must contain {list_to_string_2(pos_words, "pos")} and must not contain {list_to_string_2(neg_words, "neg")}.'
                if sum(nums) == 2:
                    return f'the document must contain {list_to_string_2(words, "pos")}.'
            elif len(data) == 1:
                if 1 in nums :
                    return f'the document must contain {list_to_string_2(words, "pos")}.'
                return f'the document must {self.text_formatter("not", tc=9, underline=True)} contain {list_to_string_2(words, "neg")}.'

        def get_best_rule(data):
            """
            Gets the best combination of shared words in paths
            this words are not shared within ALL rules
            """
            def get_apriori(path_dict):
                def dict_item_count(dict_list):
                    res = []
                    for d in dict_list:
                        dd = self.dict_to_list(d)

                        res.append(dd)
                    return res

                path_dict = dict_item_count(path_dict)
                association_rules = apriori(path_dict)
                association_results = list(association_rules)
                res = []
                for item in association_results:
                    if len(list(item[0])) > 1:
                        res.append(list(item[0]))

                return res

            rules = get_apriori(data)
            nums = []
            for rule in rules:
                keys = [x[:-1] for x in rule]
                vals = [int(x[-1]) for x in rule]

                num = 0
                for d in data:
                    if all(d.get(k, '-') == v for k, v in zip(keys, vals)):
                        num += 1
                nums.append(num * len(rule))
            maxnum = max(nums)
            maxind = nums.index(maxnum)
            return rules[maxind], maxnum

        def divide_rules(kind):

            if kind == 'add':
                dicts = self.path_dict_add
            elif kind == 'del':
                dicts = self.path_dict_del
            elif kind == 'still':
                dicts = self.path_dict_still
            rule, _ = get_best_rule(dicts)

            matched = []
            rest = []
            for d in dicts:
                keys = [x[:-1] for x in rule]
                vals = [int(x[-1]) for x in rule]
                if all(d.get(k, '-') == v for k, v in zip(keys, vals)):
                    matched.append(d)
                else:
                    rest.append(d)
            return ((matched, rule), rest)


        ############ Creating class name ########################
        print('=' * 70)
        print(self.label)
        print('=' * 70)
        print('\n')

        ############ Creating Colored bar ########################
        for kind in ['add', 'del', 'still']:

            if kind == 'add':
                color = 155 #green
                title_thing = 'added'
                title = 'The model now uses the following classification rules for this class:'
                # tot_num: total number of paths which ends as "kind"
                tot_num = len(self.path_dict_add)
                Ns = self.N_add

            elif kind == 'del':
                color = 1 #red
                title_thing = 'deleted'
                title = 'The model is not using the following classification rules anymore:'
                tot_num = len(self.path_dict_del)
                Ns = self.N_del

            elif kind == 'still':
                color = 220 #yellow  ---> 4: blue
                title_thing = 'unchanged'
                title = 'The following classification rules are unchanged throughout time:'
                tot_num = len(self.path_dict_still)
                Ns = self.N_still

            title = self.text_formatter(title, bc=color)
            if tot_num > 0:
                print(title)


            ############ stating total and used paths ########################
            if tot_num == 0:

                colored_title_thing = self.text_formatter(title_thing, bc=color)
                print(f"There are no '{colored_title_thing}' classification rules.\n ")
                continue

            rule_rules = 'rule'
            if tot_num > 1:
                rule_rules = 'rules'

            to_print = f'This class has {tot_num} {title_thing.lower()} classification {rule_rules}.'

            # if not all of them are used for the classification:
            if sum(Ns) < tot_num:
                to_print = to_print[:-1]
                is_are = 'are'
                if sum(Ns) == 1:
                    is_are = 'is'
                to_print += f', but only {sum(Ns)} {is_are} used to classify the {int(self.threshold*100)}% of the items.'

            print(self.wrapper.fill(to_print))

            print()
            #print(f'used_paths--> {used_paths}')
            #print(f'Ns --> {Ns}')

            if kind == 'add':
                used_paths = self.used_paths_add
            elif kind == 'del':
                used_paths = self.used_paths_del
            elif kind == 'still':
                used_paths = self.used_paths_still
            if sum(Ns) <= 4:
                num_list = []
                for i, item in enumerate(used_paths):
                    ze, on = agg_0_1(item)
                    num_list.append((i, len(ze) + len(on)))
                num_list = list(sorted(num_list, key=lambda x: x[1]))
                num_list = [x[0] for x in num_list]
                #############################
                ###########listing###########
                #############################
                #print(f'num_list: {num_list}')
                #print(f'Ns: {Ns}')
                for ni in num_list:
                    item = used_paths[ni]
                    rem = self.text_formatter('Having', tc=10)
                    rem2 = self.text_formatter('not', tc=9, underline=True)

                    ze, on = agg_0_1(item)
                    if 'XXXX' in list_to_string(on, add_feature=False):
                        if 'XXXX' not in list_to_string(ze, add_feature=False):
                            if len(ze) > 1:
                                iii = 'are'
                            else:
                                iii = 'is'
                            print(
                                f' - If there {iii} {rem2} {list_to_string_2(ze ,"pos")}.')
                            #print('here1')
                    elif 'XXXX' not in list_to_string(on, add_feature=False):
                        if 'XXXX' not in list_to_string(ze, add_feature=False):
                            if len(ze) > 1:
                                iii = 'are'
                            else:
                                iii = 'is'

                            print(
                                f' - {rem} {list_to_string_2(on, "pos")} but {rem2} {list_to_string_2(ze ,"neg")}.')
                            #print('here2')
                        else:
                            print(
                                f' - {rem} {list_to_string_2(on, "pos")}.')
                print()

            # Complex case
            else:
                matched, rest = divide_rules(kind)
                has_rule = matched[0]
                rule = matched[1]

                # ------ Getting the number of rules with some shared part inside
                num_list = []
                for i, item in enumerate(has_rule):
                    ze, on = agg_0_1(item)
                    num_list.append((i, len(ze) + len(on)))
                num_list = list(sorted(num_list, key=lambda x: x[1]))
                num_list = [x[0] for x in num_list]
                sag = 0
                for ni in num_list:
                    item = has_rule[ni]
                    if Ns[ni]:
                        sag += 1
                # ------------------------------------------------

                # number of paths w/o shared parts
                # some_num = sum(Ns[-(tot_num - len(matched[0])):])
                # paths used for classification
                final_remaining = sum(Ns)

                vaz = 1
                if sag > 1 and sum(Ns) > sag:
                    print(f'\nOut of these {sum(Ns)} classification rules, {sag} share the following criteria:')
                else:
                    vaz = 0

                keys_to_remove = []
                #print(f'rem --> {final_remaining}')
                #print(f'sag --> {sag}')

                if final_remaining != sag:
                    #print('XXXXXXXXXX')
                    if sag != 1:
                        print(f'{rules_to_shared(rule)}')
                        #print(f'rules --> {rule}')


                        #print(f'to be removed--> {rule} ')
                        # removing the common words from the paths
                        keys_to_remove = [x[:-1] for x in rule]


                #########################################
                ######## State remainder of shared rules#########
                #########################################
                #print(f'has_rule: {has_rule}')
                if len(has_rule) > 1 : #and sag != 1:

                    ###############
                    # DELETED STUFF
                    ###############

                    if sag != 1 :
                        #print(f'final remaining --> {final_remaining}')
                        if sum(Ns) > 2:
                            msg = 'In addition, one of the following must hold:'
                        elif sum(Ns) == 2:
                            msg = 'In addition, the following must hold:'
                        if vaz == 1:
                            print(msg)

                    num_list = []
                    for i, item in enumerate(has_rule):
                        ze, on = agg_0_1(item)
                        num_list.append((i, len(ze) + len(on)))
                    num_list = list(sorted(num_list, key=lambda x: x[1]))
                    num_list = [x[0] for x in num_list]
                    #############################
                    ###########listing###########
                    #############################
                    #print(f'num_list: {num_list}')
                    #print(f'Ns: {Ns}')
                    for ni in num_list:
                        item = has_rule[ni]

                        # checking for the frequency
                        if Ns[ni] and vaz == 1:
                            #print('here')


                            for k in keys_to_remove:
                                self.remove_key(item, k)
                                #print(f'removed --> {k}')
                            rem = self.text_formatter('Having', tc=10)
                            rem2 = self.text_formatter('not', tc=9, underline=True)

                            ze, on = agg_0_1(item)
                            if 'XXXX' in list_to_string(on, add_feature=False):
                                if 'XXXX' not in list_to_string(ze, add_feature=False):
                                    if len(ze) > 1:
                                        iii = 'are'
                                    else:
                                        iii = 'is'
                                    print(
                                        f' - If there {iii} {rem2} {list_to_string_2(ze ,"pos")}.')
                                    #print('here1')
                            elif 'XXXX' not in list_to_string(on, add_feature=False):
                                if 'XXXX' not in list_to_string(ze, add_feature=False):
                                    if len(ze) > 1:
                                        iii = 'are'
                                    else:
                                        iii = 'is'

                                    print(
                                        f' - {rem} {list_to_string_2(on, "pos")} but {rem2} {list_to_string_2(ze ,"neg")}.')
                                    #print('here2')
                                else:
                                    print(
                                        f' - {rem} {list_to_string_2(on, "pos")}.')
                                    #print('here3')


                    remaining = tot_num - sag
                    final_remaining = sum(Ns[-(remaining):])


                    paths_to_list = rest
                    # if there is nothing shared
                    if sag != 1:
                        if final_remaining == 1:
                            rem = '\nHere is the remaining rule:'
                            print(f'{rem}')
                        elif final_remaining > 1:
                            print('\nHere are the remaining rules:')

                    if sag == 1:
                        paths_to_list = used_paths
                    #print(f'rest --> {rest}')
                    #print(f'sag --> {sag}')
                    to_show = sum(Ns) - sag
                    num_list = []
                    for i, item in enumerate(paths_to_list[:to_show]):

                        ze, on = agg_0_1(item)
                        num_list.append((i, len(ze) + len(on)))
                    num_list = list(sorted(num_list, key=lambda x: x[1]))
                    num_list = [x[0] for x in num_list]

                    #############################
                    ###########listing###########
                    #############################
                    for ni in num_list:
                        item = paths_to_list[ni]
                        #print(item)
                        #print(f'Ns --> {Ns}')
                        # checking for the frequency
                        #print(f'Ns --> {Ns}')
                        #print(f'remaining --> {remaining}')
                        #print(f'ni --> {ni}')
                        #if Ns[-(remaining):][ni]:


                        rem = self.text_formatter('Having', tc=10)
                        rem2 = self.text_formatter('not', tc=9, underline=True)

                        ze, on = agg_0_1(item)
                        if 'XXXX' in list_to_string(on, add_feature=False):
                            if 'XXXX' not in list_to_string(ze, add_feature=False):
                                if len(ze) > 1:
                                    iii = 'are'
                                else:
                                    iii = 'is'
                                print(
                                    f' - If there {iii} {rem2} {list_to_string_2(ze ,"neg")}.')
                                #print('here1')
                        elif 'XXXX' not in list_to_string(on, add_feature=False):
                            if 'XXXX' not in list_to_string(ze, add_feature=False):
                                if len(ze) > 1:
                                    iii = 'are'
                                else:
                                    iii = 'is'

                                print(
                                    f' - {rem} {list_to_string_2(on, "pos")} if there {iii} {rem2} {list_to_string_2(ze ,"neg")}.')
                                #print('here2')
                            else:
                                print(
                                    f' - {rem} {list_to_string_2(on, "pos")}.')
                    print()
