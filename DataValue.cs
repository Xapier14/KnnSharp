using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Xapier14.KnnSharp
{
    public enum DataValueType
    {
        Number,
        String
    }
    public class DataValue<T> where T : INumber<T>
    {
        public DataValueType ValueType { get; }
        private readonly T? _number;
        private readonly string? _string;

        public DataValue(object data)
        {
            switch (data)
            {
                case null:
                    throw new ArgumentNullException(nameof(data));
                case INumber<T>:
                    ValueType = DataValueType.Number;
                    _number = (T)data;
                    return;
                default:
                    ValueType = DataValueType.String;
                    _string = data as string ?? data.ToString();
                    break;
            }
        }

        public T GetNumber()
            => _number!;

        public string GetString()
            => _string ?? _number?.ToString() ?? "n/a";

        public static implicit operator T(DataValue<T> value)
            => value.GetNumber();

        public static implicit operator DataValue<T>(T value)
            => new(value);

        public static implicit operator DataValue<T>(string value)
            => new(value);

        public override string ToString()
        {
            return GetString();
        }
    }
}
