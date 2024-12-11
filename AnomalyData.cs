using Microsoft.ML.Data;

namespace Lab4;

partial class Program
{
    public class AnomalyData
    {
        [LoadColumn(0)] public float X { get; set; }
        [LoadColumn(1)] public float Y { get; set; }
        [LoadColumn(2)] public float Z { get; set; }
        [LoadColumn(3)] public float Sensor1 { get; set; }
        [LoadColumn(4)] public float Sensor2 { get; set; }
        [LoadColumn(5)] public float Sensor3 { get; set; }
        [LoadColumn(6)] public float Sensor4 { get; set; }
        [LoadColumn(7)] public float Label { get; set; }
    }
}
