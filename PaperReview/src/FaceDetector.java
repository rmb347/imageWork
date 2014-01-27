import java.util.List;


public class FaceDetector {
	
	public static void main(String args[])
	{
	FaceDetector <DetectedFace ,FImage > fd = new
			HaarCascadeDetector(40);
			List <DetectedFace > faces =
			fd.detectFaces( Transforms.
			calculateIntensity(frame));
			for( DetectedFace face : faces ) {
			frame.drawShape(face.getBounds(), RGBColour.RED);
			}
	}
}
