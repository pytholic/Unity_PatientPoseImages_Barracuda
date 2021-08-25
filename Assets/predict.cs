using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using System.Linq;
using UnityEngine.UI;

public class predict : MonoBehaviour
{
    public NNModel modelSource;

    public const int IMAGE_HEIGHT = 240;
    public const int IMAGE_WIDTH = 320;
    private const float IMAGE_MEAN = 127.5f;
    private const float IMAGE_STD = 127.5f;
    const string INPUT_NAME = "X";
    const string OUTPUT_NAME = "Y";
    private string[] labels = {"head_left", "head_right"}; 
    private string prediction;
    //public TextAsset imageAsset;
    // Start is called before the first frame update
    void Start()
    {
        var image = new Texture2D(2,2);
        image.LoadImage( System.IO.File.ReadAllBytes(@"/home/trojan/Downloads/test_img2.png") );

        var flipped = FlipTexture(image);


        //image.LoadImage(imageAsset.bytes);
        // Texture2D image;
        // image = Resources.Load("test_img") as Texture2D;

        if (image) {
            Debug.Log("image found");
        } else {
            Debug.Log("image not found");
        }
        
        //byte[] pixels = image.EncodeToPNG();
        Color32[] pixels = image.GetPixels32();
        Color32[] pixels_flipped = flipped.GetPixels32();
        //Color testPixel = pixels[30000];
        Color32 testPixel = image.GetPixel(50, 50); // should be same as flipped cv2
        Color32 testPixel_flipped = flipped.GetPixel(50, 50); // should be same as original cv2
        //byte testPixel = pixels[30000];
        Debug.Log(testPixel);
        Debug.Log(testPixel_flipped);
        //Debug.Log(pixels.Length);

        var model = ModelLoader.Load(modelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        //Tensor inputTensor = TransformInput(pixels);        
        var inputTensor = TransformInput(pixels_flipped, IMAGE_HEIGHT, IMAGE_WIDTH);
        //Debug.Log(inputTensor[0, 0]);

        //print("Input:" + inputTensor);
        //var testInput = inputTensor.shape;
        //Debug.Log(testInput);
        worker.Execute(inputTensor);

        Tensor output = worker.PeekOutput(OUTPUT_NAME);
        //var output = worker.PeekOutput();
        //print("This is the output: " + output);
        //print("This is the output: " + (output[0] < 0.5? 0 : 1));
        List<float> temp = output.ToReadOnlyArray().ToList();

        print("Network predictions:" + temp[0] + ", " + temp[1]);
        
        float max = temp.Max();
		//int[] index = output.ArgMax();
        int index = temp.IndexOf(max);

        prediction = labels[index];
        print("Prediction is: " + prediction);

        inputTensor.Dispose();
        output.Dispose();
        worker.Dispose();
    }

    public static Tensor TransformInput(Color32[] pic, int width, int height)
    {
        float[] floatValues = new float[width * height * 3];

        for (int i = 0; i < pic.Length; ++i)
        {
            var color = pic[i];

            floatValues[i * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
        }

        Debug.Log(floatValues.Min() + ", " + floatValues.Max());
        return new Tensor(1, height, width, 3, floatValues);
    }

//transform from 0-255 to -1 to 1
	// Tensor TransformInput(byte[] pixels){
	// 	float[] transformedPixels = new float[pixels.Length];

	// 	for (int i = 0; i < pixels.Length; i++){
	// 		transformedPixels[i] = (pixels[i] - 127f) / 128f;
	// 	}
	// 	return new Tensor(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3, transformedPixels);
	// }

    Texture2D FlipTexture(Texture2D original){
            Texture2D flipped = new Texture2D(original.width,original.height);
         
            int xN = original.width;
            int yN = original.height;
         
           for(int i=0;i<xN;i++){
              for(int j=0;j<yN;j++){
                  flipped.SetPixel(i, yN-j-1, original.GetPixel(i,j));
              }
           }
            flipped.Apply();
         
            return flipped;
        }
}
//     //transform from 0-255 to -1 to 1
// 	Tensor TransformInput(byte[] pixels){
// 		float[] transformedPixels = new float[pixels.Length];

// 		for (int i = 0; i < pixels.Length; i++){
// 			transformedPixels[i] = (pixels[i] - 127f) / 128f;
// 		}
// 		return new Tensor(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3, transformedPixels);
// 	}
// }