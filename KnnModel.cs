using System.Data;
using System.Numerics;

namespace Xapier14.KnnSharp
{
    public enum DistanceFormula
    {
        Euclidean,
        Manhattan
    }
    public class KnnClassifierModel<T> where T : INumber<T>
    {
        private DataSet<T>? _dataSet;
        private int _dataSetClassIndex;

        private readonly Dictionary<DistanceFormula, Func<DataPoint<T>, DataPoint<T>, double>> _distanceFormulas = new();

        public KnnClassifierModel()
        {
            _distanceFormulas.Add(DistanceFormula.Euclidean, CalculateEuclideanDistance);
            _distanceFormulas.Add(DistanceFormula.Manhattan, CalculateManhattanDistance);
        }

        public DistanceFormula DistanceFormula { get; set; } = DistanceFormula.Euclidean;
        public uint KValue { get; set; } = 3;

        public double TrainAndTest(DataSet<T> dataSet, int classColumnIndex = -1, double testPercentage = 0.4, int seed = 0)
        {
            var n = dataSet.Count();
            if (dataSet.FieldCount < 3)
                throw new ArgumentException("DataSet must at least have three fields or more.", nameof(dataSet));

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

        public double TrainAndTest(DataSet<T> trainingDataSet, DataSet<T> testingDataSet, int classColumnIndex = -1)
            => TrainAndTest(trainingDataSet, testingDataSet, classColumnIndex, classColumnIndex);

        public double TrainAndTest(DataSet<T> trainingDataSet, DataSet<T> testingDataSet, int trainingDataSetClassColumnIndex, int testingDataSetClassColumnIndex)
        {
            Train(trainingDataSet, trainingDataSetClassColumnIndex);
            return Test(testingDataSet, testingDataSetClassColumnIndex);
        }

        public void Train(DataSet<T> dataSet, int classColumnIndex = -1)
        {
            if (dataSet.FieldCount < 2)
                throw new ArgumentException("DataSet must at least have two fields or more.", nameof(dataSet));

            if (classColumnIndex == -1)
                classColumnIndex = dataSet.FieldCount - 1;

            _dataSet = new DataSet<T>(dataSet.FieldCount);
            _dataSetClassIndex = classColumnIndex;

            foreach (var dataPoint in dataSet)
            {
                _dataSet.AddPoint(dataPoint);
            }
        }

        public double Test(DataSet<T> dataSet, int classColumnIndex = -1)
        {
            if (_dataSet is null)
                throw new InvalidOperationException("Data set is not loaded. Use Train() to load.");

            if (_dataSet.FieldCount != dataSet.FieldCount)
                throw new ArgumentException("Field count mismatch between training data and testing data.");

            if (classColumnIndex == -1)
                classColumnIndex = dataSet.FieldCount - 1;

            var correct = 0.0;
            foreach (var testingDataPoint in dataSet)
            {
                var prediction = Classify(testingDataPoint);
                var actual = testingDataPoint[classColumnIndex].GetString();
                if (prediction == actual)
                    correct++;
            }

            return correct / dataSet.Count();
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
            if (dataPoint1.Length == _dataSet!.FieldCount)
                dataPoint1 = dataPoint1.ExtractWithoutIndex(classIndex);
            if (dataPoint2.Length == _dataSet!.FieldCount)
                dataPoint2 = dataPoint2.ExtractWithoutIndex(classIndex);

            if (_distanceFormulas.TryGetValue(DistanceFormula, out var calculateFunc))
            {
                return calculateFunc(dataPoint1, dataPoint2);
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

        private static double CalculateEuclideanDistance(DataPoint<T> dataPoint1, DataPoint<T> dataPoint2)
        {
            var n = dataPoint1.Length;

            var distanceSquaredSum = 0.0;

            for (var i = 0; i < n; ++i)
            {
                if (dataPoint1[i].GetNumber() - dataPoint2[i].GetNumber() is not T difference)
                    throw new ArgumentException("Invalid data points.");

                distanceSquaredSum += (double)Convert.ChangeType(difference * difference, TypeCode.Double);
            }

            return distanceSquaredSum;
        }

        private static double CalculateManhattanDistance(DataPoint<T> dataPoint1, DataPoint<T> dataPoint2)
        {
            var n = dataPoint1.Length;

            double totalSum = 0;

            for (var i = 0; i < n; ++i)
            {
                if (dataPoint1[i].GetNumber() - dataPoint2[i].GetNumber() is not T difference)
                    throw new ArgumentException("Invalid data points.");

                totalSum += Math.Abs((double)Convert.ChangeType(difference, TypeCode.Double));
            }

            return totalSum;
        }
    }
}