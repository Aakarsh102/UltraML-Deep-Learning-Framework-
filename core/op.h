#pragma once

// Forward-declaration of the autograd graph node. The node definition is
// intentionally left out of this build; ops today do not touch it. When the
// autograd module is added, it will:
//
//   1. define AutogradNode with (inputs, backward_fn, saved_tensors),
//   2. set Tensor::grad_fn on outputs during forward passes,
//   3. invoke the same backward() functions declared in this framework,
//      passing the tensors it saved during forward.
//
// Every backward function in ultraml already takes its required "saved"
// tensors as explicit parameters, so autograd can be layered on top without
// re-implementing any kernel.

namespace ultraml {

struct AutogradNode;

} // namespace ultraml
