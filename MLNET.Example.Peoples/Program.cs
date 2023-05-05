using Microsoft.ML;
using MLNET.Example.Peoples;
using MLNET_Example_Peoples;

Console.WriteLine("ML NET Learning");

MLContext mlContext = new MLContext();

string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "people-100.csv");

IDataView dataView = mlContext.Data.LoadFromTextFile<PeopleModel>(dataPath);

//Predict data