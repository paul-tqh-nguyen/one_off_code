#!/usr/bin/python

"""

This is to figure out which option is better for a specific bupkis game situation. 
The situation is that we rolled three dice and two of them are good (i.e. two dice are either 1 or 5) and the other is bad (i.e. the third dice is some other value other than 1 or 5). 
We want to get a new hand by making sure at the end of the round we have all three dice be either 1 or 5. 
We want to know if it's better to reroll two of the dice (one of the good ones and the bad one) or just to reroll the bad one to get a new hand. 

Let's think about the math. 
    Case 1: We choose to reroll only a single dice.
        If we reroll one dice, we have a 2/6 chance of getting a new hand and a 4/6 chance of having a bupkis. This covers all the cases as 2/6+4/6=6/6=1.
        Thus, we expect to have a 1/3 chance of getting a new hand by rerolling one dice. 
    Case 2: We choose to reroll both dice.
        There are four cases we have to consider for the first reroll. At the end of the day, we can have both dice be a good, no dice be good, or only one dice is good (after the first reroll).
        Case 2-a: Both dice are good. 
            In order to have both dice be good, we have to have both dice be 1 or 5. 
            The probability that a single dice is 1 or 5 is 2/6 or 1/3. 
            Thus, the probability that case 2-a happens (i.e. both dice are good) is (1/3)*(1/3)=1/9.
        Case 2-b: No dice are good. 
            The probability that one dice is bad is 4/6 or 2/3. 
            Thus, the probability that case 2-b happens (i.e. both dice are bad) is (2/3)*(2/3)=4/9.
        Case 2-c: Only one dice is good after the first reroll. 
            Case 2-c-i: Dice A is good and Dice B is bad.
                The probability that Dice A is good is 2/6 or 1/3.
                The probability that Dice B is bad is 4/6 or 2/3. 
                Thus, the probability that case 2-c-i happens is (1/3)*(2/3)=2/9.
                We now get an additional second reroll. 
                    The probability that this second reroll is good is 2/6=1/3 and the probability that this second reroll is bad is 4/6=2/3. 
                    Case 2-c-i-A: The second reroll is good. 
                        Thus, the probability that this second reroll is good is (1/3)*(2/9)=2/27. 
                    Case 2-c-i-B: The second reroll is bad. 
                        Thus, the probability that this second reroll is bad is (2/3)*(2/9)=4/27.
                    Sanity Check: The sum of probabilities of cases 2-c-i-A and 2-c-i-B should be the probability of 2-c-i. This is true as 2/27+4/27=6/27=2/9.
            Case 2-c-ii: Dice A is bad and Dice B is good after the first reroll. 
                This is similar to case 2-c-i. We'll label cases 2-c-i-A and 2-c-i-B the same as we would in case 2-c-i. 
                Thus, the probability that case 2-c-ii happens is (1/3)*(2/3)=2/9.
            The probability of case 2-c happening is the sum of the probabilities of cases 2-c-i and 2-c-ii. 
            Thus, the probability of case 2-c happening is 4/9. 
        Sanity Check: The sum of probabilities of cases 2-a, 2-b, and 2-c should be 1. This is true as 1/9+4/9+4/9=9/9=1. 
        Thus, the probability of getting a success in case 2 is the sum of the probabilities of case 2a, 2-c-i-a, and 2-c-ii-a happening. This is 1/9+2/27+2/27=7/27.

"""

import random
import pdb
import time

def main():
    reroll_both_fail_count = 0
    reroll_both_success_count = 0
    reroll_single_fail_count = 0
    reroll_single_success_count = 0
    
    start_time = time.time()
    
    while True:
        # Reroll Both
        dice_one = random.randint(1,6)
        dice_two = random.randint(1,6)
        if sorted((dice_one,dice_two)) in ([1,5],[1,1],[5,5]): # both are good
            reroll_both_success_count+=1
        elif dice_one in [1,5] and dice_two not in [1,5]: # dice one is good but dice two is bad
            dice_two = random.randint(1,6)
            if dice_two in [1,5]: 
                reroll_both_success_count+=1
            else:
                reroll_both_fail_count+=1
        elif dice_one not in [1,5] and dice_two in [1,5]: # dice one is bad but dice two is good
            dice_one = random.randint(1,6)
            if dice_one in [1,5]: 
                reroll_both_success_count+=1
            else:
                reroll_both_fail_count+=1
        elif dice_one not in [1,5] and dice_two not in [1,5]: # both are bad
            reroll_both_fail_count+=1
        else:
            print dice_one, dice_two
            print "SOMETHING WENT HORRIBLY WRONG WITH THE DOUBLE REROLL"
            pdb.set_trace()
            exit(1)
        # Reroll One
        dice_one = random.randint(1,6)
        if dice_one in [1,5]:
            reroll_single_success_count+=1
        elif dice_one not in [1,5]:
            reroll_single_fail_count+=1
        else:
            print dice_one, dice_two
            print "SOMETHING WENT HORRIBLY WRONG WITH THE SINGLE REROLL"
            pdb.set_trace()
            exit(1)
                
        if time.time()-start_time>5:
            print "Single Reroll Success Rate: %5f %d/%d" % (float(reroll_single_success_count)/(reroll_single_success_count+reroll_single_fail_count),reroll_single_success_count,reroll_both_success_count+reroll_both_fail_count)
            print "Both Reroll Success Rate:   %5f %d/%d" % (float(reroll_both_success_count)/(reroll_both_success_count+reroll_both_fail_count),reroll_both_success_count,reroll_both_success_count+reroll_both_fail_count)
            print 
            start_time = time.time() 

if __name__ == '__main__': 
    main() 

