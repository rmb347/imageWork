import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class FakeResultFilter {

	
	static HashMap<String, HashMap<String, Integer> > id2data = new HashMap<String, HashMap<String, Integer> >();
	static HashMap<String, List<String>> id2line = new HashMap<String, List<String>>();
	static final float threshold = 0.8f;
	static final File opFile = new File("/Users/Aurnob/Desktop/cotraining files/multilabelDataFinal.csv");
	
	public FakeResultFilter() throws IOException
	{
		if(opFile.exists())
		{
			opFile.delete();
			opFile.createNewFile();
		}
	}
	
	
	public static void main(String args[])
	{
		File dataFile = new File("/Users/Aurnob/Desktop/cotraining files/multilabelData.csv");
		if (!dataFile.exists())
			return;
		FileInputStream inputStream;
		try {
			inputStream = new FileInputStream(dataFile);
			DataInputStream in = new DataInputStream(inputStream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String url;
			while ((url = br.readLine()) != null) {
				System.out.println(url);
				String[] urlLines = url.split(",");
				String id = urlLines[0];
				String shareWith = urlLines[2].trim();
				String shareHow = urlLines[3].trim();
				String tags = urlLines[4].trim();
				
				add2Map(id, shareWith+shareHow+tags);
				add2Set(id, url);
			}
			
			printFilteredCSV();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	private static void add2Set(String id, String data) {
		List<String> innerList;
		if(id2line.containsKey(id))
		{
			innerList = id2line.get(id);
			innerList.add(data);
		}
		else
		{
			innerList = new ArrayList<String>();
			innerList.add(data);
		}
		id2line.put(id, innerList);
	}
	private static void printFilteredCSV() {
		for(String id : id2data.keySet())
		{
			HashMap<String, Integer> innerMap = id2data.get(id);
			int totalCount = id2line.get(id).size();
			boolean printable = true;
			for(String entries : innerMap.keySet())
			{
				int count = innerMap.get(entries);
				float ratio = (float) count/totalCount;
				if(ratio > threshold)
				{
					printable = false;
				}
			}
			
			if(printable)
			{
				printToFile(id);
			}
		}
	}
	private static void printToFile(String id) {
		List<String> dataEntries = id2line.get(id);
		for(String entry : dataEntries)
		{
			try {
				PrintWriter fw = new PrintWriter(new FileWriter(opFile, true), true);
				fw.write(entry+"\n");
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	private static void add2Map(String id, String data) {
		HashMap<String, Integer> innerMap;
		if(id2data.containsKey(id))
		{
			innerMap = id2data.get(id);
			int count = 0;
			if(innerMap.containsKey(data))
			{
				count = innerMap.get(data);
			}
			innerMap.put(data,++count);
		}
		else
		{
			innerMap = new HashMap<String, Integer>();
			innerMap.put(data, 1);
		}
		id2data.put(id, innerMap);
	}
	
}
