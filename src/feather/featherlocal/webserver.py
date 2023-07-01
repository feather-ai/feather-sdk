from flask import Flask, render_template, send_from_directory
from feather.helpers import JsonObject
from flask_restful import Resource, Api, request
from flask_cors import CORS
import threading
import sys
import json
import os
import webbrowser
import traceback

# Disable the flask spew when it starts up
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

_web = Flask(__name__, template_folder='www/public', static_folder='www/public')
_api = Api(_web)
_cors = CORS(_web, resources={r"/v1/system/*": {"origins": "*"}})

_webRunning = False
_runner = None

_js = "https://feather-fwc-local.s3.us-east-2.amazonaws.com/main.js"

def HandleException(e):
    msg = str(e)
    stack_trace = traceback.format_exc()

    # Print the info too, so the user can debug it
    print("ERROR:", msg)
    print(stack_trace)

    return {"error":msg, "stack_trace": stack_trace}, 400

def build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

class GetSystemInfoHandler(Resource):
    # /v1/system/info
    # 
    # Get information about the currently loaded system.
    # The return value includes a "system_id" which is unique for this system.
    # Subsequent APIs need to provide the context ID
    def get(self):
        global _runner
        if _runner == None:
            return None, 404
        return _runner.get_system_info(), 200

class GetStepInputsHandler(Resource):
    # /v1/system/<string:contextId>/step/<int:stepIndex>/info
    #
    # Get information on the inputs and name of a step, by index.
    # This is only really valid if:
    #  1. It's the first step OR
    #  2. It's not the first step, but the previous step has been evaluated.
    #
    # As such the steps need to be called in order (we need to evaluate a step in order to grab it's outputs)
    def get(self, system_id, stepIndex):
        global _runner
        if _runner == None:
            return None, 404
        if _runner.context_id != system_id:
            return None, 410 # GONE

        try:
            info = _runner.get_step_inputs_info(stepIndex)
            return info, 200
        except Exception as e:
            return HandleException(e)

class RunStepHandler(Resource):
    # /v1/system/<string:contextId>/step/<int:stepIndex>
    #
    # Run evaluation for step "stepIndex".
    # We require a JSON payload of the format:
    #
    # {
    #   "step_name": [
    #       { "name": "<input_name_0>", "value": <value0>},
    #       { "name": "<input_name_N>", "value": <valueN>}
    #   ]
    # }
    def put(self, system_id, stepIndex): 
        if _runner.context_id != system_id:
            return None, 410 # GONE

        stepIndex = int(stepIndex)
        print("Running next step:", stepIndex)

        payload = request.get_json()

        # Get the step, and the input definition for the step, and parse the user provided
        # arguments. We check that the order, and names match up
        step = _runner.get_step_inputs_info(stepIndex)
        if step["name"] not in payload:
            return "No inputs found for step name '{0}'".format(step["name"]), 400

        # Can only evaluate if valid
        if step["is_valid"] == False:
            return "Attempt to run step out of sequence", 400

        raw_inputs = payload[step["name"]]
        step_inputs = step["inputs"]

        if len(step_inputs) != len(raw_inputs):
                return "Step '{0}' Wants '{1}' inputs, but '{2}' where proved".format(step["name"], len(step_inputs), len(raw_inputs))

        input_payloads = []
        for ri in raw_inputs:
            inputName = ri["name"]
            inputIndex = len(input_payloads)

            if step_inputs[inputIndex]["name"] != inputName:
                return "Input '{0}' name mistmatch. Want '{1}', got '{2}'".format(inputIndex, step_inputs[inputIndex]["name"], inputName)
            input_payloads.append(ri)

        # Finally, call the step run, passing the inputs
        try:
            outputs = _runner.run_step(stepIndex, input_payloads)
        except Exception as e:
            return HandleException(e)

        return outputs, 200

class DoPublish(Resource):
    # /v1/system/<string:contextId>/publish
    #
    # Publish the system to feather
    def put(self, system_id):
        global _runner
        if _runner == None:
            return None, 404
        if _runner.context_id != system_id:
            return None, 410 # GONE

        payload = request.get_json()
        apiKey = payload["apiKey"]

        try:
            info = _runner.publish(apiKey)
            return info, 200
        except PermissionError:
            return "Unauthorized", 401
        except Exception as e:
            return HandleException(e)

class DoPoll(Resource):
    # /v1/system/<string:contextId>/poll
    #
    # Return publish progress information
    # 200 = Publish completed
    # 400 = Publish not started
    # 206 = Publish in progress. Body contains progress information
    def get(self, system_id):
        global _runner
        if _runner == None:
            return None, 404
        if _runner.context_id != system_id:
            return None, 410 # GONE

        status, currFile, numFiles, currBytes, totalBytes = _runner.publish_progress()
        if status == 206:
            obj = JsonObject()
            obj.curr_file = currFile
            obj.total_files = numFiles
            obj.curr_file_size = totalBytes
            obj.curr_file_done = currBytes
            return json.loads(obj.toJSON()), status
        return "", status

_api.add_resource(GetSystemInfoHandler, "/v1/system/info")
_api.add_resource(GetStepInputsHandler, "/v1/system/<string:system_id>/step/<int:stepIndex>/info")
_api.add_resource(RunStepHandler, "/v1/system/<string:system_id>/step/<int:stepIndex>")
_api.add_resource(DoPublish, "/v1/system/<string:system_id>/publish")
_api.add_resource(DoPoll, "/v1/system/<string:system_id>/poll")

#----------------------------------------------------------------
# Handler for the root path. Display the start page of the system runner
# perhaps provide info on the system, etc...
#----------------------------------------------------------------
@_web.route("/")
def serve_system():
    # Display the Page for the current step
    return render_template("index.html", model=_runner.name, js_url=_js)

#----------------------------------------------------------------
# Handle a web request to serve a file off the root
#----------------------------------------------------------------
@_web.route("/<path:arg>")
def file(arg): 
    return send_from_directory(_web.static_folder, arg)

#----------------------------------------------------------------
# WebServer management API
# The webServer has 1 system bound at any one time. This is the system that
# is currently being run by the user.
# Note that the bound system may change between API calls, as such, we expose a
# system_id in the running system. The browser should detect a change of ID and reset
# the experience
#----------------------------------------------------------------

def _run_server():
    _web.run(port=5000, debug=True, use_debugger=False)

# Start the web server - we only ever start one even if exoprting multiple systems/models
def start_server(runner):
    if os.environ.get("FEATHER_SERVICE_RUNNER") != None:
        raise RuntimeError("Calling start_server from Runner")

    new_system(runner)

    global _webRunning
    if _webRunning == True:
        return
    _webRunning = True

    #threading.Timer(1, webbrowser.open_new_tab, args=["http://localhost:5000"]).start()
    
    _run_server()

# Called to bind a runnable system to the server. 
def new_system(runner):
    global _runner
    _runner = runner
