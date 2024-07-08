package nn

import (
	"testing"

	"github.com/jnaraujo/numgo/ng"
	"github.com/stretchr/testify/assert"
)

func TestNeuron(t *testing.T) {
	weights := ng.NewArray(0, 1.0)
	bias := 4.0
	n := NewNeuron(weights, bias)
	assert.InDelta(t, 0.99908, n.FeedForward(ng.NewArray(2.0, 3.0)), 0.01)
}
