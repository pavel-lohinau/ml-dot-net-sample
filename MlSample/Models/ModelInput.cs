using Microsoft.ML.Data;

namespace MlSample
{
  public class ModelInput
  {
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(1)]
    public float RevId { get; set; }

    [LoadColumn(2)]
    public string Comment { get; set; }

    [LoadColumn(3)]
    public int Year { get; set; }

    [LoadColumn(4)]
    public bool LoggedIn { get; set; }

    [LoadColumn(5)]
    public string Ns { get; set; }

    [LoadColumn(6)]
    public string Sample { get; set; }

    [LoadColumn(7)]
    public string Split { get; set; }
  }
}
