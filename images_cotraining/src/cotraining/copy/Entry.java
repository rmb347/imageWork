/*
* Project: airlDM
*
* Created on Nov 29, 2004 
*
*/

package cotraining.copy;

import java.io.Serializable;
import java.util.Comparator;


public class Entry implements Serializable, Comparable<Entry>, Comparator<Entry>
{
    public int i;
    public double d;
    
    public Entry(int i, double d)
    {
    	this.i = i;
    	this.d = d;
    }

    public Entry(Entry e)
    {
    	this.i = e.i;
    	this.d = e.d;
    }

    public void copy(Entry e)
    {
    	this.i = e.i;
    	this.d = e.d;
    }
    
    public final int compareTo(Entry e)
    {
        return ( (this.d<e.d) ? -1 : ( (this.d>e.d) ? 1 : ( (this.i<e.i) ? -1 : ( (this.i>e.i) ? 1 : 0) ) ) );
    }

	public int compare(Entry o1, Entry o2)
	{
        return ( (o1.d<o2.d) ? -1 : ( (o1.d>o2.d) ? 1 : ( (o1.i<o2.i) ? -1 : ( (o1.i>o2.i) ? 1 : 0) ) ) );
	}
}