using System.Data;
using System.Numerics;

namespace Xapier14.KnnSharp
{
    public enum DistanceFormula
    {
        Euclidean
    }
    public class KnnClassifierModel<T> where T : INumber<T>
    {
        private DataSet<T>? _dataSet;
        private int _dataSetClassIndex;

        private readonly Dictionary<DistanceFormula, Func<DataPoint<T>, DataPoint<T>, int, int, double>> _distanceFormulas = new();

        public KnnClassifierModel()
        {
            _distanceFormulas.Add(DistanceFormula.Euclidean, CalculateEuclideanDistance);
        }

        public DistanceFormula DistanceFormula { get; set; } = DistanceFormula.Euclidean;
        public uint KValue { get; set; } = 3;

        public double TrainAndTest(DataSet<T> dataSet, int classColumnIndex = -1, int k = 3, double testPercentage = 0.4, int seed = 0)
        {
            var n = dataSet.Count();
            if (dataSet.FieldCount < 3)
                throw new ArgumentException("DataSet must at least have three fields or more.", nameof(dataSet));

            // auto is last field
            if (classColumnIndex == -1)
                classColumnIndex = dataSet.FieldCount - 1;

            var testCount = (int)(dataSet.Count() * testPercentage);
            var trainingIndices = new LinkedList<int>();
            var testingIndices = new List<int>();
            var random = new Random(seed);

            for (var i = 0; i < n; ++i)
                trainingIndices.AddLast(i);

            for (var i = 0; i < testCount; ++i)
            {
                var index = random.Next(0, n);
                testingIndices.Add(index);
                trainingIndices.Remove(index);
                n--;
            }

            var trainingDataSet = new DataSet<T>(dataSet.FieldCount);
            foreach (var i in trainingIndices)
                trainingDataSet.AddPoint(dataSet.GetRow(i));
            var testingDataSet = new DataSet<T>(dataSet.FieldCount);
            foreach (var i in testingIndices)
                testingDataSet.AddPoint(dataSet.GetRow(i));

            return TrainAndTest(trainingDataSet, testingDataSet, classColumnIndex);
        }

        public double TrainAndTest(DataSet<T> trainingDataSet, DataSet<T> testingDataSet, int classColumnIndex)
            => TrainAndTest(trainingDataSet, testingDataSet, classColumnIndex, classColumnIndex);

        public double TrainAndTest(DataSet<T> trainingDataSet, DataSet<T> testingDataSet, int trainingDataSetClassColumnIndex, int testingDataSetClassColumnIndex)
        {
            if (trainingDataSet.FieldCount < 2)
                throw new ArgumentException("DataSet must at least have two fields or more.", nameof(trainingDataSet));

            if (trainingDataSet.FieldCount != testingDataSet.FieldCount)
                throw new ArgumentException("Field count mismatch between training data and testing data.");

            _dataSet = new DataSet<T>(trainingDataSet.FieldCount);
            _dataSetClassIndex = trainingDataSetClassColumnIndex;

            foreach (var trainingDataPoint in trainingDataSet)
            {
                _dataSet.AddPoint(trainingDataPoint);
            }

            var correct = 0.0;
            foreach (var testingDataPoint in testingDataSet)
            {
                var prediction = Classify(testingDataPoint);
                var actual = testingDataPoint[testingDataSetClassColumnIndex].GetString();
                if (prediction == actual)
                    correct++;
            }

            return correct / testingDataSet.Count();
        }

        public string Classify(DataPoint<T> dataPoint)
        {
            if (_dataSet == null)
                throw new InvalidOperationException("Data set must be loaded before using model.");

            var n = _dataSet.Count();
            var distances = new double[n];
            var classes = new string[n];
            for (var i = 0; i < n; ++i)
            {
                var referencePoint = (DataPoint<T>)_dataSet.GetRow(i);
                distances[i] = CalculateDistance(referencePoint, dataPoint, _dataSetClassIndex);
                classes[i] = referencePoint[_dataSetClassIndex].GetString();
            }

            Array.Sort(distances, classes);
            return PickTopKMajority(classes);
        }

        public string Classify(params T[] featureValues)
        {
            var dataPoint = new DataPoint<T>(featureValues.Length);
            for (var i = 0; i < featureValues.Length; ++i)
                dataPoint[i] = new DataValue<T>(featureValues[i]);
            return Classify(dataPoint);
        }

        private double CalculateDistance(DataPoint<T> dataPoint1, DataPoint<T> dataPoint2, int classIndex)
        {
            if (_distanceFormulas.TryGetValue(DistanceFormula, out var calculateFunc))
            {
                return calculateFunc(dataPoint1, dataPoint2, classIndex, _dataSet!.FieldCount);
            }

            throw new InvalidOperationException("DistanceFormula is invalid.");
        }

        private string PickTopKMajority(IReadOnlyList<string> array)
        {
            var counts = new Dictionary<string, int>();
            for (var i = 0; i < array.Count && i < KValue; ++i)
            {
                if (!counts.TryGetValue(array[i], out var oldCount))
                    oldCount = 0;
                oldCount++;
                counts[array[i]] = oldCount;
            }

            var maxCount = int.MinValue;
            var maxLabel = string.Empty;
            foreach (var (label, count) in counts)
            {
                if (count <= maxCount)
                    continue;
                maxLabel = label;
                maxCount = count;
            }

            return maxLabel;
        }

        private static double CalculateEuclideanDistance(DataPoint<T> dataPoint1, DataPoint<T> dataPoint2, int classIndex, int fieldCount)
        {
            if (dataPoint1.Length == fieldCount)
                dataPoint1 = dataPoint1.ExtractWithoutIndex(classIndex);
            if (dataPoint2.Length == fieldCount)
                dataPoint2 = dataPoint2.ExtractWithoutIndex(classIndex);
            var n = dataPoint1.Length;

            var distanceSquaredSum = 0.0;

            for (var i = 0; i < n; ++i)
            {
                if (dataPoint1[i].GetNumber() - dataPoint2[i].GetNumber() is not double plusMin)
                    throw new ArgumentException("Invalid data points.");

                distanceSquaredSum += plusMin * plusMin;
            }

            return distanceSquaredSum;
        }
    }
}