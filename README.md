# ITBS_Stock_Model

### Goal 
The investment model that based on the movement of investment trust companies. The algorithm screens daily stock trading to deicover sudden large amount trading from investment trusts.

### Main Document
TW_itbs_exp.py

### Steps
The model demonstrate the investment strategy that "Follow the movement of Investment Trust Companies". The program will auto-execute the following steps:

#### Collect Data
 
#### Check the Conditions
If a stock satisfies 
1. IT(Investment Trust Companies) have been untrading for a long time 
2. IT buy a lot today
3. The marketcap of the company is small

then it can be a candidate target.

### Demo
in *demo_example.py*, you can key in a date (yyyymmdd) you want to check. The program will check if there's any signal on that day.

### Some issues
1. Caution the version update for the target webpage (fix)
2. The loading time (fix)
3. Should consider foreign investment
