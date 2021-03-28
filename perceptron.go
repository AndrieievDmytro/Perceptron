package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
)

var (
	// trainFile     string = "data\\train.txt" //data file
	// testFile      string = "data\\test.txt"  //data file
	weightsString string  = "[1,3,6]"
	threshold     float64 = 67
	weightsNum    []float64
	alfa          float64 = 0.1
	output        int
	// beta          float64
	// dimension     int
)

// Activation function

type Object struct {
	Params []float64
	Name   string
	// lable  int
}
type Objects struct {
	Ob []Object
}

func init() {
	// flag.StringVar(&trainFile, "tr", trainFile, "train-file")
	// flag.StringVar(&testFile, "ts", testFile, "test-file")
	// flag.IntVar(&dimension, "d", dimension, "dimension")
	flag.StringVar(&weightsString, "w", weightsString, "weights in form [val1,val2,valN]")
	flag.Float64Var(&threshold, "t", threshold, "threshold")
	flag.Float64Var(&alfa, "a", alfa, "alfa")
}

func stringToFloatArray(value string) []float64 {
	str := value
	var numVal []float64
	err := json.Unmarshal([]byte(str), &numVal)
	if err != nil {
		log.Fatal(err)
	}
	return numVal
}

func computeOutput(input []float64) int {
	var dotProduct float64
	for i := range input {
		dotProduct += input[i] * weightsNum[i]
	}
	fmt.Println("Dot product:", dotProduct)
	if dotProduct >= threshold {
		output = 1
	} else if dotProduct < threshold {
		output = 0
	}
	return output
}

func train(input []float64, desireVal int) {
	err := desireVal - output
	for i := range weightsNum {
		weightsNum[i] += float64(err) * alfa * input[i]
	}
	fmt.Println("Weight vector: ", weightsNum)
}

func main() {
	flag.Parse()
	weightsNum = stringToFloatArray(weightsString)
	testInput := []float64{10, 5, 2}
	for {
		output = computeOutput(testInput)
		if output != 1 {
			train(testInput, 1)
		} else {
			break
		}
	}
}
