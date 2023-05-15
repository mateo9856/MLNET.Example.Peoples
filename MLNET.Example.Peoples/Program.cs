using Microsoft.ML;
using MLNET.Example.Peoples;
using MLNET_Example_Peoples;
using System.Reflection;

Console.WriteLine("ML NET Learning");

MLContext mlContext = new MLContext();

string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "..", "..", "people-100.csv");

IDataView dataView = mlContext.Data.LoadFromTextFile<PeopleModel>(dataPath, hasHeader: true, separatorChar: ',');

IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName:"Sex", outputColumnName:"Label")
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "First Name",outputColumnName: "FirstNameFeaturized"))
    .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Last Name",outputColumnName: "LastNameFeaturized"))
    .Append(mlContext.Transforms.Concatenate("Features", "FirstNameFeaturized", "LastNameFeaturized"))
    .AppendCacheCheckpoint(mlContext);

var trainingPipeline = dataPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

ITransformer model = trainingPipeline.Fit(dataView);

var engine = mlContext.Model.CreatePredictionEngine<PeopleModel, PeopleModelOutput>(model);

var modelTest = new PeopleModel()
{
    FirstName = "Jack",
    LastName = "More",
    Email = "mike123@myspace.com"
};

var pred = engine.Predict(modelTest);

Console.WriteLine(pred.Sex);