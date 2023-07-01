import ast
from os import read
from feather import component_factory, components
import sys
import json
import os
import importlib.util

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

def load_dynamic_module(name, filepath):
    print("DynamicModuleLoad:", filepath)
    spec = importlib.util.spec_from_file_location(name, filepath)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

# Executor is responsible for executing a system in 'detatched' mode - this refers to executing
# the system outside of the Python environment that is was created in, entirely from serialized data
class SystemExecutor:
    def __init__(self, id, systemSchema, code_files, rootDir):
        if os.environ.get("FEATHER_SERVICE_RUNNER") == None:
            raise RuntimeError("Illegal call to SystemExecutor")
        
        self.system = systemSchema
        self.id = id
        self.counter = 0
        self.loadedModules = []
        if rootDir not in sys.path:
            sys.path.append(rootDir)

        mainModule = [code_files[0]]
        for file in mainModule:
            filename = file
            # Remove the extension
            #if filename.lower().endswith(".py"):
            #    filename = filename[:-3]
            self.loadedModules.append(self.load_module(filename, rootDir))     

    def load_module(self, module, rootDir):
        if module in sys.modules:
            print("FTR: Loading (cached) module", module)
            return sys.modules[module]

        print("FTR: Loading module", module, "from", rootDir)
        ret = load_dynamic_module(self.id + str(self.counter), rootDir + "/" + module)
        return ret

    def unload_modules(self):
        for mod in self.loadedModules:
            if mod.__name__ in sys.modules:
                print("FTR: Unloading module", mod.__name__)
                del sys.modules[mod.__name__]
            print("FTR: Deleting module", mod.__name__)
            del mod
        del self.loadedModules

    def get_step_by_name(self, stepName):
        for step in self.system["steps"]:
            if step["name"] == stepName:
                return step
        return None

    def _run_single_step(self, stepInfo, inputs):
        # Get the function pointer from the module      
        stepName = stepInfo["name"]  
        print("FTR: Running step", stepName)
        func = getattr(self.main_module(), stepName)
        if func == None:
            raise ValueError("RunStep: Step func not found:", stepName)

        outputs = func(*inputs)

        # If a function returns only one component, we need to convert it into an iterable
        try:
            iter(outputs)
        except TypeError:
            outputs = [outputs]

        return outputs

    # Given a step JSON schema and an inputs payload in JSON format,
    # instanciate any components needed and load the correct payload in
    def prepare_inputs(self, step, input_payload):
        inputDefinitions = step["inputs"]
        numInputs = len(inputDefinitions)
        if numInputs != len(input_payload):
            raise ValueError("prepare_inputs: Step {0} wants {1} arg, but have {2} in payload".format(step["name"], numInputs, len(input_payload)))

        ret = []
        for idx in range(numInputs):
            currInputDefinition = inputDefinitions[idx]
            currInputPayload = input_payload[idx]
            if currInputDefinition["type"] == "COMPONENT":
                component = component_factory.CreateComponent(currInputDefinition)
                component.component._inject_payload(currInputPayload)
                ret.append(component)
            elif currInputDefinition["type"] == "OPAQUE":
                ret.append(components.step_opaque_input_data_adapter(idx, currInputPayload))
        return ret
        
    # Example API that runs a step to completion in a server environment 
    def run_step(self, stepName, inputData):
        if stepName == "#system":
            print("FTR: Running entire system")
            inputs = self.prepare_inputs(self.system["steps"][0], inputData)
            outputs = None
            for step in self.system["steps"]:
                outputs = self._run_single_step(step, inputs)
                inputs = outputs
            return components.step_output_adapter(outputs)
        else:
            # Lookup the step info
            stepInfo = self.get_step_by_name(stepName)
            if stepInfo == None:
                raise ValueError("RunStep: Step doesn't exist:", stepName)

            inputs = self.prepare_inputs(stepInfo, inputData)
            outputs = self._run_single_step(stepInfo, inputs)
            return components.step_output_adapter(outputs)


    def main_module(self):
        return self.loadedModules[0]

# Internal functions

class RecursiveVisitor(ast.NodeVisitor):
    """ example recursive visitor """
    indent = 0
    indentStep = 2

    def recursive(func):
        """ decorator to make visitor work recursive """
        def wrapper(self,node):
            func(self,node)
            self.indent += self.indentStep
            for child in ast.iter_child_nodes(node):
                self.visit(child)
            self.indent -= self.indentStep
        return wrapper

    def prettyPrint(self, *args):
        print(" " * self.indent, *args)

    #@recursive
    def visit_Assign(self,node):
        """ visit a Assign node and visits it recursively"""
        if type(node.value).__name__ == "Call":
            self.prettyPrint("Assign:", node.targets[0].id, "= <func>")
        elif type(node.value).__name__ == "Constant":
            self.prettyPrint("Assign:", node.targets[0].id, "=", node.value.value)
        else:
            self.prettyPrint("Assign:", node.targets[0].id, "=", type(node.value).__name__)

    @recursive
    def visit_BinOp(self, node):
        """ visit a BinOp node and visits it recursively"""
        self.prettyPrint(type(node).__name__)

    @recursive
    def visit_Call(self,node):
        """ visit a Call node and visits it recursively"""
        self.prettyPrint(type(node).__name__)

    @recursive
    def visit_Constant(self,node):
        self.prettyPrint(node.value)

    def visit_Import(self,node):
        """ visit a Import node """
        if node.names[0].asname is not None:
            self.prettyPrint("Import:", node.names[0].name, "as", node.names[0].asname)
        else:
            self.prettyPrint("Import:", node.names[0].name)

    def visit_ImportFrom(self,node):
        """ Eg. from tensorflow.keras import layers """
        self.prettyPrint("From:", node.module, "import", node.names[0].name)

    @recursive
    def visit_Lambda(self,node):
        """ visit a Function node """
        self.prettyPrint(type(node).__name__)

    @recursive
    def visit_FunctionDef(self,node):
        """ visit a Function node and visits it recursively"""
        self.prettyPrint("Function:", node.name)

    @recursive
    def visit_Module(self,node):
        """ visit a Module node and the visits recursively"""
        pass

    @recursive
    def visit_Return(self,node):
        self.prettyPrint("Return")

    @recursive
    def visiit_arguments(self, node):
        dump(node)

    def visit_alias(self, node):
        """nothing"""
        pass

    @recursive
    def generic_visit(self,node):
        """empty"""
        self.prettyPrint("Generic: ", type(node).__name__)