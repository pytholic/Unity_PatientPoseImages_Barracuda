using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Linq;
using System.Text.RegularExpressions;

public class classification_script : MonoBehaviour
{
    public NNModel modelFile;

    public const int IMAGE_HEIGHT = 240;
    public const int IMAGE_WIDTH = 320;
    private const int IMAGE_MEAN = 127;
    private const float IMAGE_STD = 127.5f;
    private const string INPUT_NAME = "X";
    private const string OUTPUT_NAME = "Y";

    private IWorker worker;
    private string[] labels = {"head_left", "head_right"};

    private Texture2D picture;


    public void Start()
    {
        var model = ModelLoader.Load(this.modelFile);
        // picture = Resources.Load("1.png") as Texture2D;
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

    }

    // private int i = 0;
    public IEnumerator Classify(Color32[] picture, System.Action<List<KeyValuePair<string, float>>> callback)
    {
        var map = new List<KeyValuePair<string, float>>();

        using (var tensor = TransformInput(picture, IMAGE_HEIGHT, IMAGE_WIDTH))
        {
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add(INPUT_NAME, tensor);  
            // var enumerator = this.worker.ExecuteAsync(inputs);

            // while (enumerator.MoveNext())
            // {
            //     i++;
            //     if (i >= 20)
            //     {
            //         i = 0;
            //         yield return null;
            //     }
            // };

            worker.Execute(inputs);
            //Execute() scheduled async job on GPU, waiting till completion
            yield return new WaitForSeconds(0.5f);

            var output = worker.PeekOutput(OUTPUT_NAME);

            for (int i = 0; i < labels.Length; i++)
            {
                map.Add(new KeyValuePair<string, float>(labels[i], output[i] * 100));
            }
        }

        callback(map.OrderByDescending(x => x.Value).ToList());
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

        return new Tensor(1, height, width, 3, floatValues);
    }
}
