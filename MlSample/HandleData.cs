using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using MlSample.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace MlSample
{
  public class HandleData
  {
    private const string LabelColumnName = "Label";
    private const string FeatureColumnName = "Features";
    private const string ArchiveName = "Model.zip";
    private const double TestFraction = 0.2;
    private readonly string ProjectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\"));

    public IEnumerable<ModelOutput> Predict(IEnumerable<ModelInput> modelInput)
    {
      var archiveDataPath = Path.Combine(Environment.CurrentDirectory, HandleData.ArchiveName);

      var mLContext = new MLContext();
      ITransformer trainedModel;

      if (File.Exists(archiveDataPath) == false)
      {
        trainedModel = this.CreateTrainedModel(mLContext);
      }
      else
      {
        trainedModel = this.LoadTrainedModel(mLContext);
      }

      return this.UseModelWithBatchItems(mLContext, trainedModel, modelInput);
    }

    private ITransformer CreateTrainedModel(MLContext mLContext)
    {
      var trainDataPath = $"{this.ProjectDirectory}\\Data\\Wiki.txt";
      var dataView = mLContext.Data.LoadFromTextFile<ModelInput>(trainDataPath, hasHeader: false);

      var testData = this.LoadData(mLContext, dataView);

      ITransformer trainedModel = this.BuildAndTrainModel(mLContext, testData.TrainSet);

      this.Evaluate(mLContext, trainedModel, testData.TestSet);

      this.SaveModel(mLContext, trainedModel, dataView);

      return trainedModel;
    }

    private TrainTestData LoadData(MLContext mLContext, IDataView dataView)
    {
      var splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: HandleData.TestFraction);

      return splitDataView;
    }

    private ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
    {
      var estimator = mLContext.Transforms.Text.FeaturizeText(outputColumnName: HandleData.FeatureColumnName, inputColumnName: nameof(ModelInput.Comment))
        .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: HandleData.LabelColumnName, featureColumnName: HandleData.FeatureColumnName));

      var model = estimator.Fit(splitTrainSet);

      return model;
    }

    private void Evaluate(MLContext mLContext, ITransformer trainedModel, IDataView splitTestSet)
    {
      var predictions = trainedModel.Transform(splitTestSet);
      var metrics = mLContext.BinaryClassification.Evaluate(predictions, HandleData.LabelColumnName);

      Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
      Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
      Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    }

    private IEnumerable<ModelOutput> UseModelWithBatchItems(MLContext mLContext, ITransformer trainedModel, IEnumerable<ModelInput> modelInput)
    {
      var batchComments = mLContext.Data.LoadFromEnumerable(modelInput);

      var predictions = trainedModel.Transform(batchComments);

      var predictedResults = mLContext.Data.CreateEnumerable<ModelOutput>(predictions, reuseRowObject: false);

      return predictedResults;
    }

    private void SaveModel(MLContext mLContext, ITransformer model, IDataView dataView)
    {
      mLContext.Model.Save(model, dataView.Schema, HandleData.ArchiveName);
    }

    private ITransformer LoadTrainedModel(MLContext mLContext)
    {
      return mLContext.Model.Load(HandleData.ArchiveName, out var modelSchema);
    }
  }
}