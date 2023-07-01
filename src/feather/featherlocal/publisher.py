import base64
from io import BufferedReader, DEFAULT_BUFFER_SIZE, RawIOBase, SEEK_SET, SEEK_END
import json
import tempfile
import requests
import os
from slugify import slugify
from feather import helpers

class TrackedBufferedReader(BufferedReader):
    def __init__(self, raw: RawIOBase, buffer_size=DEFAULT_BUFFER_SIZE,
                 callback=None):

        raw.seek(0, SEEK_END)
        self.file_size = raw.tell()
        self.progress = 0
        raw.seek(0, SEEK_SET)

        self._callback = callback
        BufferedReader.__init__(self, raw, buffer_size=buffer_size)
        self._cb()

    def read(self, size: int) -> bytes:
        self.progress = self.progress + size
        self._cb()
        return super().read(size)

    def read1(self, size: int) -> bytes:
        self.progress = self.progress + size
        self._cb()
        return super().read1(size)

    def _cb(self):
        if self.progress > self.file_size:
            self.progress = self.file_size # Don't know why requests is trying to read beyond file size...
        self._callback(self.file_size, self.progress)

class FileData:
    def __init__(self, filename=None, filetype=None, jsonObject=None):
        if jsonObject != None:
            self.filename = jsonObject["filename"]
            self.filetype = jsonObject["filetype"]
            self.data = jsonObject["data"]
        else:
            if filename == None or filetype == None:
                raise ValueError("FileData - need jsonObject or filename+filetype")
            self.filename = filename
            self.filetype = filetype
            with open(filename, "rb") as fp:
                self.data = base64.b64encode(fp.read()).decode('utf-8')
            if self.data == None:
                raise ValueError("Cannot load file:", filename)

class Bundle:
    def __init__(self, code_files=[], model_files=[]):
        # TODO: Validate the filenames - needs to be relative folder, and .py extension
        self.code = code_files
        self.models = model_files
        # TODO Hash the files and store the hashes - so we can check that they haven't changed during the upload, as
        # uploads of large files may take a looong time
        self.total_files = len(code_files) + len(model_files)
        self.files_uploaded = 0
        self.upload_curr_file_bytes_done = 0
        self.upload_curr_file_size = 0

    def progress_callback(self, size, progress):
        self.upload_curr_file_size = size
        self.upload_curr_file_bytes_done = progress

    def do_publish(self, name, version, system_schema, server_url, api_key):
        if os.environ.get("FEATHER_SERVICE_RUNNER") != None:
            raise RuntimeError("Calling do_publish from Runner")
        
        # Get the upload URLs from the server
        reqBody = helpers.JsonObject()
        reqBody.name = name
        reqBody.slug = slugify(name)
        reqBody.version = version
        reqBody.schema = system_schema
        reqBody.files = []

        cwd = os.getcwd()
        for file in self.code:
            info = helpers.JsonObject()
            info.filename = helpers.CleanRelativePath(cwd, file)
            info.filetype = "python"
            reqBody.files.append(info)

        for file in self.models:
            info = helpers.JsonObject()
            info.filename = helpers.CleanRelativePath(cwd, file)
            info.filetype = "model"
            reqBody.files.append(info)

        for f in reqBody.files:
            print("Preparing to upload: {0}".format(f))

        body = reqBody.toJSON(pretty=False)
        headers = {
            'X-FEATHER-API-KEY': api_key,
            'Content-Type': 'application/json'
            }
        print("PREPARE TO PUBLISH:", body)
        prep = requests.put(server_url + "/v1/api/system/preparePublish", data=body, headers=headers)
        if prep.status_code == 500: # Service return  500 (son't ask me why lol) for API Key failure
            raise PermissionError()
        if prep.status_code != 200:
            raise SystemError("Error starting publish. Server response: ", prep.status_code, prep.content)

        uploadData = json.loads(prep.content)
        uid = uploadData["id"]

        print("Publish Request ID=", uid)
        file_upload_headers = {
            "Content-Type":"application/octet-stream",
        }
        # Upload the files
        for upload in uploadData["files"]:
            print("FileUpload=", upload)
            filename = upload["filename"]
            uploadUrl = upload["uploadUrl"]
            with open(filename, "rb") as fp:
                trackedFp = TrackedBufferedReader(fp.raw, callback=self.progress_callback)
                print("- Uploading:", filename)
                r = requests.put(url=uploadUrl, data=trackedFp, headers=file_upload_headers)
                if r.status_code != 200:
                    raise SystemError("Error uploading file:", filename, "to URL", uploadUrl, "error=", r.content)

                self.files_uploaded = self.files_uploaded + 1

        # Complete the request
        r = requests.put(server_url + "/v1/api/system/completePublish", json={'id': uid}, headers=headers)
        if r.status_code != 200:
            raise SystemError("Error completing publish. ", r.content)
        print("Publish Complete")
        return json.loads(r.content.decode("utf-8"))
