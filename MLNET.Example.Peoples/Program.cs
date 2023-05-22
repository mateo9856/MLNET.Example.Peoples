using Microsoft.ML;
using MLNET.Example.Peoples;
using MLNET_Example_Peoples;
using Newtonsoft.Json.Linq;
using System.Reflection;

Console.WriteLine("ML NET Learning");

MLContext mlContext = new MLContext();

string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "people-100.csv");

IDataView dataView = mlContext.Data.LoadFromTextFile<PeopleModel>(dataPath, hasHeader: true, separatorChar: ',');

IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Sex", outputColumnName: "Label")
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "First Name", outputColumnName: "FirstNameFeaturized"))
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Last Name", outputColumnName: "LastNameFeaturized"))
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Job Title", outputColumnName: "JobTitleFeaturized"))
    .Append(mlContext.Transforms.Concatenate("Features", "FirstNameFeaturized", "LastNameFeaturized", "JobTitleFeaturized"));

var trainingPipeline = dataPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = trainingPipeline.Fit(dataView);

var engine = mlContext.Model.CreatePredictionEngine<PeopleModel, PeopleModelOutput>(model);

var modelTest = new PeopleModel()
{
    FirstName = "Matt",
    LastName = "More",
    JobTitle = "Builder"
};

var pred = engine.Predict(modelTest);

Console.WriteLine(pred.Sex);