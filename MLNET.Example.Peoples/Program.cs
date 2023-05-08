using Microsoft.ML;
using MLNET.Example.Peoples;
using MLNET_Example_Peoples;

Console.WriteLine("ML NET Learning");

MLContext mlContext = new MLContext();

string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "people-100.csv");

IDataView dataView = mlContext.Data.LoadFromTextFile<PeopleModel>(dataPath);

IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Conversion.MapValueToKey("Sex", "Label")
    .Append(mlContext.Transforms.Text.FeaturizeText(nameof(PeopleModel.FirstName), "FirstNameFeaturized"))
    .Append(mlContext.Transforms.Text.FeaturizeText(nameof(PeopleModel.LastName), "LastNameFeaturized"))
    .Append(mlContext.Transforms.Text.FeaturizeText(nameof(PeopleModel.Email), "EmailFeaturized"))
    .Append(mlContext.Transforms.Concatenate("Features", "FirstNameFeaturized", "LastNameFeaturized", "EmailFeaturized"))
    .AppendCacheCheckpoint(mlContext);

var trainingPipeline = dataPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

ITransformer model = trainingPipeline.Fit(dataView);

var engine = mlContext.Model.CreatePredictionEngine<PeopleModel, PeopleModelOutput>(model);