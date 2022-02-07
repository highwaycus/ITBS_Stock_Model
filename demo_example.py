# This is Demo example to show how the model works

import datetime
from TW_itbs_exp import ITBS


def check_insert_date_valid(day):
    day_ = datetime.datetime.strptime(str(int(day)), '%Y%m%d')
    if day_.weekday() > 4:
        print('The date you entered is probably weekend. Are you sure you want to continue? (y/n)')
        ans = input()
        if ans == 'y':
            return True
        else:
            return False
    return True


########################################
########################################
if __name__ == '__main__':
    print('Enter a Date (YYYYMMDD):')
    print('(Hint: try enter 20210118)')
    insert_date = input()
    day_valid = check_insert_date_valid(insert_date)
    if day_valid:
        demo = ITBS()
        demo.daily_main(today=insert_date)
    else:
        print('End')
