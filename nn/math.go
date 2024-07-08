package nn

import (
	"math"

	"github.com/jnaraujo/numgo/ng"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func MSELoss(yTrue, yPred ng.Array[float64]) float64 {
	return yTrue.Sub(yPred).Power(2).Mean()
}
