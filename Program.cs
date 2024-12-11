using System;
using System.Linq;
using Microsoft.ML;

namespace Lab4;

partial class Program
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        string dataPath = "../../../data.csv";
        var data = mlContext.Data.LoadFromTextFile<AnomalyData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',');

        var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainingData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        var dataPipeline = mlContext.Transforms.Concatenate(
                outputColumnName: "Features",
                "X", "Y", "Z", "Sensor1", "Sensor2", "Sensor3", "Sensor4");

        var trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(
            featureColumnName: "Features", rank: 3);

        var trainingPipeline = dataPipeline.Append(trainer);

        Console.WriteLine("Training model...");
        var model = trainingPipeline.Fit(trainingData);

        Console.WriteLine("Evaluating model...");
        var predictions = model.Transform(testData);
        var metrics = mlContext.AnomalyDetection.Evaluate(
            predictions, scoreColumnName: "Score", labelColumnName: "Label");

        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");

        var predictor = mlContext.Model.CreatePredictionEngine<AnomalyData, AnomalyPrediction>(model);

        var sample = new AnomalyData
        {
            X = 6.912997186032634f,
            Y = 11.5186980127206f,
            Z = 15.47185522229428f,
            Sensor1 = 0.0f,
            Sensor2 = 0.0f,
            Sensor3 = 0.0f,
            Sensor4 = 1.0f,
            Label = 0.0f
        };

        var prediction = predictor.Predict(sample);

        Console.WriteLine($"Anomaly: {(prediction.PredictedLabel ? "Yes" : "No")}");
        Console.WriteLine($"Score: {prediction.Score}");

        string modelPath = "output.zip";
        mlContext.Model.Save(model, trainingData.Schema, modelPath);
        Console.WriteLine($"\nМодель збережено у {modelPath}");
    }
}
