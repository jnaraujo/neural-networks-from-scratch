package nn

import "github.com/jnaraujo/numgo/ng"

type Neuron struct {
	weights ng.Array[float64]
	bias    float64
}

func NewNeuron(weights ng.Array[float64], bias float64) *Neuron {
	return &Neuron{
		weights: weights,
		bias:    bias,
	}
}

func (n *Neuron) FeedForward(inputs ng.Array[float64]) float64 {
	return Sigmoid(n.weights.Dot(inputs) + n.bias)
}
