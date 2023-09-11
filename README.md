﻿# KNN Sharp

## Getting Started
1. Create a console project.
1. Download and add the nuget package to your project.
1. Download the [sample dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) and place it on the root of your project and in the same directory as your build artifact.
1. Replace the contents of `Program.cs` with:
   ```csharp
    using Xapier14.KnnSharp;

    // load the data set
    var knn = new KnnClassifierModel<double>();
    var dataSet = DataSet<double>.LoadFromCsvFile("iris.data");
    dataSet.SetDataLabels("Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Class");
    var accuracy = knn.TrainAndTest(dataSet);
    Console.WriteLine("KNN data set loaded. Accuracy: {0}%", accuracy * 100.0);

    // test the data set interactively
    while (true)
    {
        Console.WriteLine("-Predict-");
        Console.Write("Sepal Length: ");
        var sepalLength = double.Parse(Console.ReadLine()!);
        Console.Write("Sepal Width: ");
        var sepalWidth = double.Parse(Console.ReadLine()!);
        Console.Write("Petal Length: ");
        var petalLength = double.Parse(Console.ReadLine()!);
        Console.Write("Petal Length: ");
        var petalWidth = double.Parse(Console.ReadLine()!);
        Console.WriteLine("Predicted Classification: {0}", knn.Classify(sepalLength, sepalWidth, petalLength, petalWidth));
        Console.WriteLine();
    }
   ```
1. Run the project.