import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

public class redirect {

	private static final String IMAGETAG_TEMPLATE =  "<p><img src= alt=\"some_text\" width=\"304\" height=\"228\" /></p>";

	public static void main(String args[]) {
		String location = System.getProperty("DOWNLOAD_LOCATION",
				"/Users/Aurnob/Desktop/cotraining files/mechanic turk/");
		File filename = new File(location + "sample.csv");
		File fileout = new File(location + "sample_static.csv");
		File htmltemplate = new File(location + "templ.txt");
		File htmlOut = new File(location + "templ_out.html");
		PrintWriter out = null;
		try {
			out = new PrintWriter(fileout);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		PrintWriter out1 = null;
		try {
			out1 = new PrintWriter(htmlOut);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		if (filename.exists()) {
			try {
				FileInputStream inputStream = new FileInputStream(filename);
				DataInputStream in = new DataInputStream(inputStream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String url = null;
				String stUrl = null;
				String htmlText = gethtmlText(htmltemplate);
				while ((url = br.readLine()) != null) {
					System.out.println(url);
					String[] starr = url.split(",");
					stUrl = starr[1];
					writeOut(getImageTag(stUrl) + htmlText,  out1);
					stUrl = null;
				}
				out.close();
				out1.close();
				in.close();
				br.close();
			}

			catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	private static void writeOut(String htmlText, PrintWriter out) {
		out.println(htmlText);
		out.println("");
		
	}
	private static String getImageTag(String stUrl) {
		int index = IMAGETAG_TEMPLATE.indexOf("src=");
		String imageTag = IMAGETAG_TEMPLATE.substring(0, index+4) + "\"" + stUrl + "\"" + IMAGETAG_TEMPLATE.substring(index+5, IMAGETAG_TEMPLATE.length()); 
		return imageTag;
	}
	private static String gethtmlText(File file)
	{
		String str = "" ;
		if (file.exists()) {
			try {
				FileInputStream inputStream = new FileInputStream(file);
				DataInputStream in = new DataInputStream(inputStream);
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
				String line = null;
				while ((line = br.readLine()) != null) {
					str += line;
				}
				in.close();
				br.close();
			}

			catch (IOException e) {
				e.printStackTrace();
			}
		}
		return str;
	}

}
