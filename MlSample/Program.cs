using System;
using System.Collections.Generic;

namespace MlSample
{
  class Program
  {
    static void Main(string[] args)
    {
      HandleData handleData = new HandleData();

      IEnumerable<ModelInput> model = new[]
      {
        new ModelInput { Comment = "You are fucking idiot"},
        new ModelInput { Comment = "I love you"}
      };

      var data = handleData.Predict(model);

      foreach (var item in data)
      {
        Console.WriteLine($"Comment: {item.Comment}\nIsToxic: {item.Prediction}\nScore: {item.Score}");
        Console.WriteLine();
      }

      Console.ReadLine();
    }
  }
}