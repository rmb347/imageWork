package cotraining.copy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class SumofN {

    public static void main(String[] args) {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution. */
        Scanner sc = new Scanner(System.in);
        int N = sc.nextInt();
        Double K = sc.nextDouble();
        Double [] arr = new Double[N];
        for(int i = 0 ;i < N ; i++)
        	arr[i] = sc.nextDouble();
        
        Arrays.sort(arr);
        
        double [] [] table  = new double[(int) (K+1)][N];      
        
        for(Double i = arr[0] ; i < K+1 ; i++)
        {
        	for( int j = 0 ; j < N ; j++)
        	{
        		if(arr[j] <= i)
        		{
        			double diff = i - arr[j];
        			if(diff > 0)
        			{
        				table[diff][]
        			}
        		}
        			
        	}
        }
        
        List<Double> list = new ArrayList<Double>();
        
        for(int i = 0; i < arr.length && arr[i] < K ; i++)
        {
        	list.add(arr[i]);
        }
        
        System.out.println( findRecursiveBest(K, list ));
        
    }
    
    static int findRecursiveBest(Double K, List<Double> list )
    {
    	int max = 0;
        for(int i = 0; i < list.size() && list.get(i) < K ; i++)
        {
        	List<Double> clonedList = new ArrayList<Double>();
        	clonedList.addAll(list);
        	clonedList.remove(i);
        	int num = findRecursiveBest( (K-list.get(i)), clonedList) + 1;
        	if(num > max)
        	{
        		max = num;
        	}
        }
        return max;
    }
    
    
}