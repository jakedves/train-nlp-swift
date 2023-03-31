import CreateML
import Foundation

// dataset I used this with: https://www.kaggle.com/datasets/parulpandey/emotion-dataset
// license: CCO 1.0, Public Domain Dedication (approved for free cultural works)

// example: "/Users/johnappleseed/Downloads/my_dataset.csv"
let csvPath = "CHANGE_THIS"

// example: "/Users/johnappleseed/Downloads/"
let modelOutputPath = "CHANGE_THIS"

let csv = URL(fileURLWithPath: csvPath)
let modelDestination = URL(fileURLWithPath: modelOutputPath)

var parsingOptions = MLDataTable.ParsingOptions()

parsingOptions.containsHeader = true
parsingOptions.delimiter = ","
parsingOptions.lineTerminator = "\n"

var data = try MLDataTable(contentsOf: csv, options: parsingOptions)

// this code changes labels from Int (unsupported) to String
// my csv file was:
// text,label
// example sentence to record,1
// another sentence to record,3
//
// but we need:
// example sentence to record,anger
// another sentence to record,joy
let labelColumnName = "label"
let stringLabelColumnName = "stringLabel"
let textColumnName = "text"

let labelsDictionary: [Int : String] = [
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear"
]

let stringLabelColumn = data[labelColumnName].map { label -> String in
    return labelsDictionary[label] ?? "unknown"
}

// replace the integer column with the string classification column
data.removeColumn(named: labelColumnName)
data.addColumn(stringLabelColumn, named: stringLabelColumnName)

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 8)

let emotionClassifier = try MLTextClassifier(trainingData: trainingData,
                                             textColumn: textColumnName,
                                             labelColumn: stringLabelColumnName)

let trainingAccuracy = (1.0 - emotionClassifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - emotionClassifier.validationMetrics.classificationError) * 100

let evaluationMetrics = emotionClassifier.evaluation(on: testingData, textColumn: textColumnName, labelColumn: stringLabelColumnName)
let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

print("Training Accuracy:   \(trainingAccuracy)")
print("Validation Accuracy: \(validationAccuracy)")
print("Evaluation Accuracy: \(evaluationAccuracy)")

let metadata = MLModelMetadata(author: "YOUR NAME HERE",
                               shortDescription: "A model trained to identify emotion of first person sentences",
                               version: "1.0")

try emotionClassifier.write(to: modelDestination, metadata: metadata)
