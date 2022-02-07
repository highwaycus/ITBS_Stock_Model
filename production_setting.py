# Basic Functions Tools

def sub_process_bar(j, total_step):
    str_ = '>' * (50 * j // total_step) + ' ' * (50 - 50 * j // total_step)
    sys.stdout.write('\r[' + str_ + '][%s%%]' % (round(100 * j / total_step, 2)))
    sys.stdout.flush()
    j += 1
    return j
    

def tw_path_setting(collapse='daily', weight='equal', auto_proba=True):
    if collapse in ['日', '週', '月']:
        chinese_collapse = collapse
        collapse = collapse_translate_ch_to_en(collapse=chinese_collapse)
    else:
        chinese_collapse = collapse_translate(collapse)
    collapse = collapse.capitalize()
    use_public = False
    path_loading = 'tw_data/'
    path_saving = 'tw_data/TWStockSignal{}Proba/'.format(collapse)
    path_combination = 'tw_data/TWStockGroupSignal{}ProbaWeight{}/'.format(collapse, weight.capitalize())
    path_visual = path_combination[:-1] + 'Visual/'
    for path in [path_loading, path_saving, path_combination, path_visual, path_etf]:
        try:
            os.makedirs(path)
        except:
            pass
    return path_loading, path_saving, path_combination, path_visual
    

def display_setting():
    np.set_printoptions(precision=5, suppress=True, linewidth=150)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 500)
