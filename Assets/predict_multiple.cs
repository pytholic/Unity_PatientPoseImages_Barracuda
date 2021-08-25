using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using UnityEngine.UI;
using System;

public class predict_multiple : MonoBehaviour
{
    public NNModel modelSource;

    public const int IMAGE_HEIGHT = 240;
    public const int IMAGE_WIDTH = 320;

    const string INPUT_NAME = "X";
    const string OUTPUT_NAME = "Y";

    private const float IMAGE_MEAN = 127.5f;
    private const float IMAGE_STD = 127.5f;

    private string[] labels = {"head_left", "head_right"}; 
    private string prediction;
    public string path = "/home/trojan/unity/head_location_onnx_testing/Images";
    public string[] image_list;
    
    private IWorker worker;

    void Start()
    {

        var model = ModelLoader.Load(modelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        image_list = System.IO.Directory.GetFiles(@path, "*.png");
        int cnt = image_list.Length;
        // for (int i=0; i<cnt; i++)
        // {
        //     Debug.Log(image_list[i]);
        // }

        Debug.Log($"Total images: {cnt}");

        for (int i=0; i<cnt; i++){
            var image = new Texture2D(2,2);
            image.LoadImage(System.IO.File.ReadAllBytes(@image_list[i]));
            var image_flipped = FlipTexture(image);
            Color32[] pixels = image_flipped.GetPixels32();

            // Debug.Log(image.height + ", " + image.width);
            // Debug.Log(pixels.Length);
            var input = TransformInput(pixels, IMAGE_HEIGHT, IMAGE_WIDTH);
            var output = worker.Execute(input).PeekOutput();

            float[] temp = output.ToReadOnlyArray();
            //print("Network raw predictions:" + temp[0] + ", " + temp[1]);

            var index = output.ArgMax()[0];
            //Debug.Log($"Image was recognised as class number: {index}");

            prediction = labels[index];
            print($"Prediction for {i} is {prediction}");

            input.Dispose();
            output.Dispose();
            
        }
        worker.Dispose();
    }

    public static Tensor TransformInput(Color32[] pic, int width, int height){
        float[] floatValues = new float[width * height * 3];

        for (int i = 0; i < pic.Length; ++i){
            var color = pic[i];

            floatValues[i * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
            }

        //Debug.Log(floatValues.Min() + ", " + floatValues.Max());
        return new Tensor(1, height, width, 3, floatValues);
        }

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