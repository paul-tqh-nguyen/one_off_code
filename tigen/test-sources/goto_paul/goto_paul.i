
/*************************************************************
 * 
 * Ideal test cases should test when input is 1 and 3 
 * and not 2 since we can never have that.
 * 
 *************************************************************/
 
int test(int input) {
    int ans = 0;
    goto L1;
    goto L2;
    
L_final:
    if (input == 3)
        ans = 3;
    return ans;
    
L1: 
    if (input == 1)
        ans = 1;
    goto L_final;
    
L2: 
    if (input == 2)
        ans = 2;
    goto L_final;
    
} 


