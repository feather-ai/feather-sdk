from feather import components
from feather.helpers import JsonObject, dump
from feather.featherlocal import publisher
from typing import Callable, List
import uuid
import os
import json
import inspect

# Special sentinel function to grab the last step's outputs


def _runner_final_step(*inputs):
    return []

# -------------------------------------------------------------------------------------------------
# Class holding information about a step during evaluation, including all inputs and outputs
# -------------------------------------------------------------------------------------------------


class StepInfo:
    MODE_INPUTS = "in"
    MODE_OUTPUTS = "out"

    def __init__(self, step: Callable, inputs: List, isStart, isEnd=False):
        self.func: Callable = step
        if self.func.__name__ == _runner_final_step.__name__:
            self.name = "#final_step"
        else:
            self.name = self.func.__name__
        self.inputs = inputs
        self.isEnd = isEnd
        self.isStart = isStart
        self.title = None
        self.description = None
        if hasattr(step, "_ftr_title"):
            self.title = step._ftr_title
        if hasattr(step, "_ftr_description"):
            self.description = step._ftr_description

        self.outputs = []
        self.schema: JsonObject = None
        self.evaluated = False

    def _serialize_inputs(self, outputSchema):
        rawArgs = inspect.getargspec(self.func)
        argNames = rawArgs.args
        if rawArgs.varargs != None:
            argNames.append(rawArgs.varargs)

        if self.name == "#final_step":
            argNames = ["*"] * len(self.inputs)

        if len(argNames) != len(self.inputs):
            raise RuntimeError("Step '{0}' expects {1} argument but {2} were found during serialization".format(
                self.name, len(argNames), len(self.inputs)))

        for idx in range(len(self.inputs)):
            stepInput = self.inputs[idx]
            inputSchema = JsonObject()
            inputSchema.name = argNames[idx]

            if isinstance(stepInput, components.FeatherComponent):
                stepInput.component._serialize_schema(inputSchema)
                payloadSchema = stepInput.component._get_payload_schema()
                inputSchema.schema = json.loads(payloadSchema)
            else:
                inputSchema.type = "OPAQUE"
            outputSchema.inputs.append(inputSchema)

    def _serialize_ouptuts(self, outputSchema):
        for idx in range(len(self.inputs)):
            stepInput = self.inputs[idx]
            info = JsonObject()

            if isinstance(stepInput, components.FeatherComponent):
                stepInput.component._serialize_schema(info)
            else:
                info.type = "OPAQUE"
            outputSchema.append(info)

    # -------------------------------------------------------
    # Serialize this step info it's full schema
    # -------------------------------------------------------
    def serialize(self, final, mode):
        if mode == StepInfo.MODE_INPUTS:
            self.schema = JsonObject()
            self.schema.name = self.name
            self.schema.title = self.title
            self.schema.description = self.description
            self.schema.inputs = []
            self.schema.outputs = []
            if final == False:
                if self.inputs != [] or self.isStart == True:       # If our inputs are defined, or it's the first step, then valid
                    self.schema.is_valid = True
                else:
                    self.schema.is_valid = False
                    return self.schema

            self._serialize_inputs(self.schema)
        elif mode == StepInfo.MODE_OUTPUTS:
            self.schema = []
            self._serialize_ouptuts(self.schema)
        else:
            raise RuntimeError("Unknown serialization mode {0}".format(mode))

        return self.schema

# -------------------------------------------------------------------------------------------------
# The system runner provides an API for stepping forwards or backwards through
# a system.
# -------------------------------------------------------------------------------------------------


class SystemRunner:
    def __init__(self, name, description, bootstrap: Callable, steps: List[Callable], file_bundle: publisher.Bundle):
        self.steps: List[StepInfo] = []
        self.currentStep = 0
        self.schema: JsonObject = None
        self.deploy = False  # True when the system should be  deployed
        self.finalOutputs = []
        self.version = "1.0.0"
        self.name = name
        self.description = description
        self.context_id = str(uuid.uuid4())
        self.file_bundle = file_bundle
        self.publish_in_progress = False
        self.publish_complete = False

        isStart = True
        for step in steps:
            self.steps.append(StepInfo(step, inputs=[], isStart=isStart))
            isStart = False

        # Append the final "output" step
        self.steps.append(StepInfo(_runner_final_step, [], isStart=isStart, isEnd=True))

        self._run_bootstrap(bootstrap)

    # -------------------------------------------------------
    # Once the system  has finished running, we've collected all the info on
    # the inputs and outputs. Serialize it all so we can save it for the service
    # -------------------------------------------------------
    def serialize(self, final):
        self.schema: JsonObject = JsonObject()
        self.schema.version = self.version
        self.schema.steps = []
        self.schema.num_steps = len(self.steps) - 1 # -1 since final_step is not a real step

        for step in self.steps:
            if step.func != _runner_final_step:  # Don't serialize the sentinel step
                self.schema.steps.append(step.serialize(final=final, mode=StepInfo.MODE_INPUTS))
            else:
                self.schema.final_outputs = step.serialize(final=final, mode=StepInfo.MODE_OUTPUTS)
                
        self.schema.steps[0].static_data = self.bootstrap_output_data

    # -------------------------------------------------------
    # Return high level info about the system
    # -------------------------------------------------------

    def get_system_info(self) -> str:
        ret = JsonObject()
        ret.num_steps = len(self.steps) - 1
        ret.name = self.name
        ret.description = self.description
        ret.system_id = self.context_id # output field is system_id not context_id to make UI simpler. BUT it is NOT the system ID!!
        return json.loads(ret.toJSON(pretty=False))

    # -------------------------------------------------------
    # Return information about the inputs for a specific step
    # -------------------------------------------------------
    def get_step_inputs_info(self, stepIndex: int) -> str:
        if stepIndex < 0:
            raise ValueError("get_step_inputs_info: Invalid step index", stepIndex, "Must be >=0")
        if stepIndex >= len(self.steps):
            raise ValueError("get_step_inputs_info: Invalid step index", stepIndex, "Must be < maxSteps")

        step = self.steps[stepIndex]
        step.serialize(final=False, mode=StepInfo.MODE_INPUTS)

        if stepIndex == 0:
            step.schema.static_data = self.bootstrap_output_data
        return json.loads(step.schema.toJSON(pretty=False))

    # -------------------------------------------------------
    # Publish a fully evaluated and serialized system to the feather service
    # -------------------------------------------------------
    def publish(self, api_key: str):
        if self.publish_complete == True:
            raise RuntimeError("Publish already done")
        if self.publish_in_progress == True:
            raise RuntimeError("Publish already in progress")

        if self.file_bundle == None or type(self.file_bundle) != publisher.Bundle:
            raise ValueError(
                "A file_bundle needs to be provided in order to publish. The bundle should contain all your code and models needed to run your system")

        for step in self.steps:
            if step.evaluated == False and step.isEnd == False:
                raise RuntimeError(
                    "Error during publish - not all steps were evaluated! You can only publish once all steps have evaluated successfully")

        self.publish_in_progress = True

        try:
            # final serialize
            self.serialize(final=True)

            baseUrl = "https://dev.feather-works.net" #"https://cn5n0ztv5d.execute-api.us-east-2.amazonaws.com/dev"
            if "FEATHER_PUBLISH_BASE_URL" in os.environ:
                baseUrl = os.environ.get("FEATHER_PUBLISH_BASE_URL")

            ret = self.file_bundle.do_publish(name=self.name, version="0.0.0",
                                        system_schema=self.schema.toJSON(), server_url=baseUrl, api_key=api_key)
            self.publish_complete = True
            return ret
        except Exception as e:
            self.publish_in_progress = False
            raise e

    # Returns upload porgress information:
    #
    def publish_progress(self):
        if self.publish_complete:
            return (200, None, None, None, None)

        if self.publish_in_progress == False:
            return (400, None, None, None, None)

        # Return 206 (partial content) for upload still in progress
        return (206, self.file_bundle.files_uploaded, self.file_bundle.total_files, self.file_bundle.upload_curr_file_bytes_done, self.file_bundle.upload_curr_file_size)

    # -------------------------------------------------------
    # Get a specific step by index
    # -------------------------------------------------------
    def get_step(self, index) -> StepInfo:
        if index < 0 or index >= len(self.steps):
            raise RuntimeError("Invalid step")

        return self.steps[index]

    # -------------------------------------------------------
    # Called to run (evaluate) step at the specified index
    # -------------------------------------------------------
    def run_step(self, stepIndex: int, inputPayloads: str):
        if stepIndex < 0:
            raise ValueError("change_step: Invalid step index", stepIndex, "Must be >=0")
        if stepIndex >= len(self.steps):
            raise ValueError("change_step: Invalid step index", stepIndex, "Must be < max steps")

        if stepIndex < self.currentStep:
            return self._prev_step()
        elif stepIndex == self.currentStep:
            return self._next_step(inputPayloads)
        else:
            raise ValueError("change_step: Attempt to jump >1 step at once. stepIndex: {}, self.currentStep: {}".format(
                stepIndex, self.currentStep))

    # -------------------------------------------------------
    # Return true if we're at the end of the system
    # -------------------------------------------------------
    def finished(self) -> bool:
        return self.currentStep == len(self.steps)

    # -------------------------------------------------------
    # Helper function which runs a system from start to finish. Should not be used in the webservice.
    # -------------------------------------------------------
    def run_to_completion(self):
        # Bootstrap has already run, so prepare them as inputs for step0
        input_payloads = self._convert_outputs_to_inputs(self.bootstrap_output_data, self.steps[0])

        self.currentStep = 0
        while self.finished() == False:
            nextStepIdx = self.currentStep + 1
            currentStep = self.steps[self.currentStep]

            print("Running step", self.currentStep, currentStep.func.__name__)
            output_data = self._next_step(input_payloads)
            #  If we have a valid next step, prepare the inputs
            if nextStepIdx < len(self.steps):
                input_payloads = self._convert_outputs_to_inputs(output_data, self.steps[nextStepIdx])

    # -------------------------------------------------------
    # Helper function that takes outputs as generated from a step, and packages them
    # into the JSON format expected for inputs into the next step. This way the
    # step has no idea that the inputs were not from an external source
    # -------------------------------------------------------
    def _convert_outputs_to_inputs(self, output_data, nextStep):
        nextStep.serialize(final=False, mode=StepInfo.MODE_INPUTS)
        inputSchema = nextStep.schema

        # From the raw outputs of the previous step, we need to generate the JSON payload so the
        # Next step can run.
        nextInputs = []
        for inputIdx, ni in enumerate(inputSchema.inputs):
            nextInput = {}
            nextInput["name"] = ni.name
            if ni.type == "COMPONENT":
                for k in output_data[inputIdx]:
                    nextInput[k] = output_data[inputIdx][k]
            else:
                nextInput["value"] = output_data[inputIdx]
            nextInputs.append(nextInput)
        return nextInputs

    # -------------------------------------------------------
    # Run user provided, system bootstrap function
    # -------------------------------------------------------
    def _run_bootstrap(self, func):
        outputs, output_data = self._run_func("bootstrap", func, [])
        self.steps[0].inputs = outputs
        self.bootstrap_output_data = output_data

    # -------------------------------------------------------
    # Actually invoke a user provided func, passing in inputs and ferrying out the outputs
    # -------------------------------------------------------
    def _run_func(self, name, func, inputs):
        try:
            outputs = func(*inputs)

            # If a function returns only one component, we need to convert it into an iterable
            try:
                iter(outputs)
            except TypeError:
                outputs = [outputs]

            if isinstance(outputs, tuple):
                outputs = list(outputs)

            return outputs, components.step_output_adapter(outputs)
        except Exception as e:
            raise e

    # -------------------------------------------------------
    # Perform the work to move to the next step
    #
    # input_payloads should be = [ {id=<component_id>, payload=<payload_for_component>} ]
    # -------------------------------------------------------
    def _next_step(self, input_payloads):
        # Run the current step first, and grab the outputs.
        step = self.steps[self.currentStep]
        components.step_input_adapter(step.inputs, input_payloads)

        try:
            outputs, output_data = self._run_func(step.name, step.func, step.inputs)
            step.outputs = outputs
            step.output_data = output_data
            step.evaluated = True

            # TODO Warn if outputs is a Tensor
        except TypeError as e:
            raise e
        except Exception as e:
            #raise e
            raise RuntimeError("Next Step failed for:", step.func, "with error=", e)

        # For the send to last step, grab the outputs and promote them to
        # system outputs. This is because the last step is our internal sentinel step which
        # doesn't output anything
        if self.currentStep == len(self.steps) - 2:
            self.finalOutputs = step.outputs if step.outputs != None else []

        self.currentStep = self.currentStep + 1

        # If we're not at the end...
        if self.currentStep < len(self.steps):
            # Outputs of current step become inputs to next step
            nextStep = self.steps[self.currentStep]
            nextStep.inputs = step.outputs if step.outputs != None else []

        return step.output_data

    # -------------------------------------------------------
    # Perform the work to move to the previous step
    # -------------------------------------------------------
    def _prev_step(self) -> bool:
        self.currentStep = self.currentStep - 1
        return self.steps[self.currentStep].output_data
