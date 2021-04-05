package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

var (
	trainFile       string = "data\\iris_training.txt" //data file
	testFile        string = "data\\iris_test.txt"     //data file
	weightsString   string
	threshold       float64
	weightsNum      []float64
	alfa            float64
	desireAccurancy float64
)

type Object struct {
	Params []float64
	Name   string
	Lable  int
	Output int
}
type Objects struct {
	Obj []Object
}

func init() {
	flag.StringVar(&trainFile, "tr", trainFile, "train-file")
	flag.StringVar(&testFile, "ts", testFile, "test-file")
	flag.Float64Var(&desireAccurancy, "acc", desireAccurancy, "Desire accurancy")
	flag.StringVar(&weightsString, "w", weightsString, "weights in form [val1,val2,valN]")
	flag.Float64Var(&threshold, "t", threshold, "threshold")
	flag.Float64Var(&alfa, "a", alfa, "alfa")
}

func readCsv(path string) ([][]string, error) {
	dataFile, err := os.OpenFile(path, os.O_RDONLY, 0666)
	if err != nil {
		return nil, err
	}
	defer dataFile.Close()
	if err == nil {
		// Reading text from file
		buf, rerr := ioutil.ReadAll(dataFile)
		if rerr != nil {
			fmt.Println("Read CSV error: " + rerr.Error())
		}

		// Parsing from comma-separated text
		r := csv.NewReader(strings.NewReader(string(buf)))
		records, err := r.ReadAll()
		if err != nil {
			fmt.Println("Parse CSV error: " + err.Error())
		}
		return records, nil
	}
	return nil, err
}

func convertStrArrayToJson(records [][]string) string {
	// Converting from array of string to JSON
	jsonData := ""
	strNum := 0
	dimension := len(records[1]) - 1
	for _, record := range records {
		wrongStr := false
		if len(record) < dimension || len(record) > dimension+1 {
			fmt.Println("Wrong parameters count")
			wrongStr = true
		}
		// var objName string
		objName := record[len(record)-1] // Cutting Object name
		record = record[:len(record)-1]
		strNum++
		stringArray := "["                // Opening sq bracket
		for _, arrField := range record { // Filling string representation of array
			_, err := strconv.ParseFloat(arrField, 64)
			if err != nil {
				fmt.Println("Wrong parameters type in string: ", strNum)
				wrongStr = true
			}
			stringArray += arrField + ","
		}
		stringArray = stringArray[:len(stringArray)-1] // Removing last ',' character
		stringArray += "]"                             // Closing sq bracket
		if !wrongStr {
			jsonData += "{ \"name\": \"" + objName + "\", \"params\":" + stringArray + ", \"Distance\": [] }," // Converting to JSON
		}
	}
	jsonData = "[" + jsonData[:len(jsonData)-1] + "]"
	return jsonData

}

func (obj *Objects) readData(path string) {
	records, err := readCsv(path)
	if err == nil {
		jsonData := convertStrArrayToJson(records)
		json.Unmarshal([]byte(jsonData), &obj.Obj)
	} else {
		fmt.Println("Read CSV error: " + err.Error())
	}
}

func stringToFloatArray(value string) {
	str := value
	err := json.Unmarshal([]byte(str), &weightsNum)
	if err != nil {
		log.Fatal(err)
	}
}

func (obj *Object) computeOutput() {
	var dotProduct float64
	for i := range obj.Params {
		dotProduct += obj.Params[i] * weightsNum[i]
	}
	// fmt.Println("Dot product:", dotProduct)
	if dotProduct >= threshold {
		obj.Output = 1
	} else {
		obj.Output = 0
	}
}

func (obj *Object) train() {
	err := obj.Lable - obj.Output
	for i := range weightsNum {
		weightsNum[i] += float64(err) * alfa * obj.Params[i]
	}
	threshold += float64(err) * alfa * -1

	// fmt.Println("Weight vector: ", weightsNum, "threashold: ", threshold)
}

func (objects *Objects) assignLabels(name string) {
	for i, obj := range objects.Obj {
		if name == obj.Name {
			objects.Obj[i].Lable = 0
		} else {
			objects.Obj[i].Lable = 1
		}
	}
}

func getResult() {
	tr := new(Objects)
	tr.readData(trainFile)
	ts := new(Objects)
	ts.readData(testFile)
	chkName := tr.Obj[0].Name
	tr.assignLabels(chkName)
	ts.assignLabels(chkName)
	accuracy := 0.0
	for iter := 0; accuracy < desireAccurancy && iter < 1000000; iter++ {
		counter := len(ts.Obj)
		// Train perceptron
		for _, obj_train := range tr.Obj {
			obj_train.computeOutput()
			if obj_train.Lable != obj_train.Output {
				obj_train.train()
			}
		}
		// fmt.Println("Weight vector: ", weightsNum, "threashold: ", threshold)
		// Calculate accurancy in test data
		for _, obj_test := range ts.Obj {
			obj_test.computeOutput()
			if obj_test.Output != obj_test.Lable {
				counter--
			}
		}
		accuracy = (float64(counter) / float64(len(ts.Obj))) * 100.0
		fmt.Println("Accurancy [", iter, "]", accuracy, "%")
	}
}

func main() {
	flag.Parse()
	stringToFloatArray(weightsString)
	fmt.Println(weightsNum)
	getResult()
}
