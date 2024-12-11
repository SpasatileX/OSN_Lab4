using Microsoft.ML.Data;

namespace Lab4;

partial class Program
{
    public class AnomalyPrediction
    {
        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        [ColumnName("Score")] public float Score { get; set; }
    }
}
