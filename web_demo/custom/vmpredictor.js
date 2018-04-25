/*
  ===============LICENSE_START=======================================================
  Acumos Apache-2.0
  ===================================================================================
  Copyright (C) 2017-2018 AT&T Intellectual Property & Tech Mahindra. All rights reserved.
  ===================================================================================
  This Acumos software file is distributed by AT&T and Tech Mahindra
  under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  This file is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ===============LICENSE_END=========================================================
*/
/**
 vmpredictor.js - angular backend for interaction with vm predictor demo

 E. Zavesky 8/8/17 adapted for Acumos
 */

"use strict";

/*
 * Control for selecting different date range
 */

angular.module('vmpredictor', ['ui.bootstrap-slider', 'ui.bootstrap'])
.controller('DatepickerPopupCtrl', function ($scope) {
  $scope.today = function() {
    $scope.dtLast = new Date();
    $scope.dtFirst = new Date();
    $scope.dtFirst.setDate($scope.dtLast.getDate() - 31);
  };
  $scope.fixed = function() {
    $scope.dtLast = new Date(2017, 3, 14);
    $scope.dtFirst = new Date(2017, 0, 1);
  }
  $scope.fixed(); // $scope.today()

  $scope.clear = function() {
    $scope.dtFirst = null;
    $scope.dtLast = null;
  };

  $scope.inlineOptions = {
    customClass: getDayClass,
    minDate: new Date(),
    showWeeks: true
  };

  $scope.dateOptions = {
    dateDisabled: disabled,
    formatYear: 'yy',
    maxDate: new Date(2020, 5, 22),
    minDate: new Date(2010, 0, 1),
    startingDay: 0
  };

  $scope.gen = {
      core: 50,
  };


  // Disable weekend selection
  function disabled(data) {
    var date = data.date,
      mode = data.mode;
    return mode === 'day' && (date.getDay() === 0 || date.getDay() === 6);
  }

  $scope.open1 = function() {
    $scope.popup1.opened = true;
  };

  $scope.open2 = function() {
    $scope.popup2.opened = true;
  };

  $scope.setDate = function(year, month, day) {
    $scope.dtFirst = new Date(year, month, day);
  };

  $scope.formats = ['dd-MMMM-yyyy', 'yyyy/MM/dd', 'dd.MM.yyyy', 'shortDate'];
  $scope.format = $scope.formats[0];
  $scope.altInputFormats = ['M!/d!/yyyy'];

  $scope.popup1 = {
    opened: false
  };

  $scope.popup2 = {
    opened: false
  };


  function getDayClass(data) {
    var date = data.date,
      mode = data.mode;
    if (mode === 'day') {
      var dayToCheck = new Date(date).setHours(0,0,0,0);

      for (var i = 0; i < $scope.events.length; i++) {
        var currentDay = new Date($scope.events[i].date).setHours(0,0,0,0);

        if (dayToCheck === currentDay) {
          return $scope.events[i].status;
        }
      }
    }

    return '';
  }

})
/*
 * Control for live VM interaction
 */
.controller('ProtoSim', function ($scope) {
  $scope.gen = {};

  // ---- method for plotting/generating data ---------------------------

  // simple example: http://jsfiddle.net/AbM9z/1/
  function sinFunc(x, y, top, ampl, sinDivide, cosDivide) {
    return {
            x   : x + ampl * Math.sin(top / sinDivide),
            y   : (top / 100 < 0.65) ? y + 2 : 1 + y + ampl * Math.cos(top / cosDivide)
        };
    }


  function vm_generate(val_mean, val_std, num_sample) {
    $scope.gen.data = [];
    $scope.gen.data_idx = 0;
    if (!num_sample)
        num_sample = 4*24*7;

    var sinDivide = 4*(15*60000);   //every hour?
    var cosDivide = sinDivide/3;    //every 8 hours; NOTE: set equal for a circle
    var last = { x:val_mean, y:50, amplitude:val_std };
    var dt = new Date();
    dt.setMinutes(dt.getMinutes() - dt.getMinutes()%15);
    for (var i=0; i<num_sample; i++) {
        var next = sinFunc(last.x, last.y, dt.getTime(), last.amplitude, sinDivide, cosDivide);
        $scope.gen.data.push({"date":dt, "name":"reference", "ref":i, "value":Math.round(next.x*1000)/1000});
        dt = new Date(dt.getTime() + 15*60000); //update by fifteen minutes
        $.extend(last, next);
    }

    /*
    console.log($scope.gen.data);
    var line = d3.line()
        .x(function(d) { return x(d.date); })
        .y(function(d) { return y(d.value); });
    line.context(context)($scope.gen.data);
    */
  }

  function protobuf_load(pathProto, forceSelect) {
    protobuf.load(pathProto, function(err, root) {
        if (err) {
            console.log("[protobuf]: Error!: "+err);
            throw err;
        }
        var domSelect = $("#protoMethod");
        var numMethods = domSelect.children().length;
        $.each(root.nested, function(namePackage, objPackage) {    // walk all
            if ('Model' in objPackage && 'methods' in objPackage.Model) {    // walk to model and functions...
                var typeSummary = {'root':root, 'methods':{} };
                $.each(objPackage.Model.methods, function(nameMethod, objMethod) {  // walk methods
                    typeSummary['methods'][nameMethod] = {};
                    typeSummary['methods'][nameMethod]['typeIn'] = namePackage+'.'+objMethod.requestType;
                    typeSummary['methods'][nameMethod]['typeOut'] = namePackage+'.'+objMethod.responseType;
                    typeSummary['methods'][nameMethod]['service'] = namePackage+'.'+nameMethod;

                    //create HTML object as well
                    var namePretty = namePackage+"."+nameMethod;
                    var domOpt = $("<option />").attr("value", namePretty).text(
                        nameMethod+ " (input: "+objMethod.requestType
                        +", output: "+objMethod.responseType+")");
                    if (numMethods==0) {    // first method discovery
                        domSelect.append($("<option />").attr("value","").text("(disabled, not loaded)")); //add 'disabled'
                    }
                    if (forceSelect) {
                        domOpt.attr("selected", 1);
                    }
                    domSelect.append(domOpt);
                    numMethods++;
                });
                $scope.gen.proto[namePackage] = typeSummary;   //save new method set
                //$("#protoContainer").show();
            }
        });
        console.log("[protobuf]: Load successful, found "+numMethods+" model methods.");
    });

  }

  $scope.gen.dl_proto = function(is_input) {

    //  https://stackoverflow.com/a/33622881
    function downloadBlob(data, fileName, mimeType) {
      //if there is no data, filename, or mime provided, make our own
      if (!data)
          data = $scope.gen.proto_in;
      if (!fileName)
          fileName = "protobuf.bin";
      if (!mimeType)
          mimeType = "application/octet-stream";

      var blob, url;
      blob = new Blob([data], {
        type: mimeType
      });
      url = window.URL.createObjectURL(blob);
      downloadURL(url, fileName, mimeType);
      setTimeout(function() {
        return window.URL.revokeObjectURL(url);
      }, 1000);
    };

    function downloadURL(data, fileName) {
      var a;
      a = document.createElement('a');
      a.href = data;
      a.download = fileName;
      document.body.appendChild(a);
      a.style = 'display: none';
      a.click();
      a.remove();
    };

    if (is_input) {
        return downloadBlob($scope.gen.proto_in, "protobuf.in.bin");
    }
    else {
        return downloadBlob($scope.gen.proto_output, "protobuf.out.bin");
    }
  }

    /**
     * post an image from the canvas to the service
     */
    $scope.gen.post_feature = function(toggle_send) {
        var sendPayload = null;
        var nameProtoMethod = $("#protoMethod option:selected").attr('value');
        var methodKeys = null;
        if (nameProtoMethod && nameProtoMethod.length) {     //valid protobuf type?
            var partsURL = $scope.gen.server.split("/");
            methodKeys = nameProtoMethod.split(".", 2);       //modified for multiple detect/pixelate models
            partsURL[partsURL.length-1] = methodKeys[1];
            $scope.gen.server = partsURL.join("/");   //rejoin with new endpoint
        }

        if (toggle_send) {  // flip the bit if required
            if ($scope.gen.trigger_active) {
                clearInterval($scope.gen.trigger_active);
                $scope.gen.trigger_active = 0;
            }
            else {
                $scope.gen.trigger_active = setInterval(vm_gen_tick, 500);
            }
        }
        var serviceURL = $scope.gen.server;
        var request = new XMLHttpRequest();     // create request to manipulate

        request.open("POST", serviceURL, true);
        $scope.gen.transmitting = true;

        //$(dstDiv).append($("<div>&nbsp;</div>").addClass('spinner'));

        //console.log("[doPostImage]: Selected method ... '"+typeInput+"'");

        // fields from .proto file at time of writing...
        //   message VmPredictorDataFrame {
        //  int64 day = 1;
        //  int64 weekday = 2;
        //  int64 hour = 3;
        //  int64 minute = 4;
        //  double hist_1D8H = 5;
        //  double hist_1D4H = 6;
        //  double hist_1D2H = 7;
        //  double hist_1D1H = 8;
        //  double hist_1D = 9;
        //  double hist_1D15m = 10;
        //  double hist_1D30m = 11;
        //  double hist_1D45m = 12;
        //  string VM_ID = 13;
        // }

        //TODO: should we always assume this is input? answer: for now, YES, always image input!
        var inputPayload = { "frames":[] };
        var val = $scope.gen.data[$scope.gen.data_idx];
        var inputFrame = { "day":val["date"].getDate(), "weekday":val["date"].getDay(), "ref":$scope.gen.data_idx,
                            "hour":val["date"].getHours(), "minute":val["date"].getMinutes(),
                         "hist_1D8H":0, "hist_1D4H":0, "hist_1D2H":0, "hist_1D1H":0,
                         "hist_1D":val["value"], "hist_1D15m":0, "hist_1D30m":0, "hist_1D45m":0 };
        $scope.gen.data_idx = ($scope.gen.data_idx+1)%$scope.gen.data.length;  //add index, loop
        inputPayload["frames"].push(val);
        $scope.gen.obj_input = $scope.gen.obj_input.concat(inputPayload['frames']);
        //console.log($scope.gen.obj_input);
        var sliceLen = $scope.gen.obj_input.length-$scope.gen.max_obj_queue;
        if (sliceLen > 0) {
            $scope.gen.obj_input = $scope.gen.obj_input.slice(sliceLen);
        }

        // ---- method for processing from a type ----
        var msgInput = $scope.gen.proto[methodKeys[0]]['root'].lookupType($scope.gen.proto[methodKeys[0]]['methods'][methodKeys[1]]['typeIn']);
        // Verify the payload if necessary (i.e. when possibly incomplete or invalid)
        var errMsg = msgInput.verify(inputPayload);
        if (errMsg) {
            console.log("[doPostImage]: Error during type verify for object input into protobuf method.");
            throw Error(errMsg);
        }
        // Create a new message
        var msgTransmit = msgInput.create(inputPayload);
        // Encode a message to an Uint8Array (browser) or Buffer (node)
        sendPayload = msgInput.encode(msgTransmit).finish();

        //downloadBlob(sendPayload, 'protobuf.bin', 'application/octet-stream');
        // NOTE: TO TEST THIS BINARY BLOB, use some command-line magic like this...
        //  protoc --decode=mMJuVapnmIbrHlZGKyuuPDXsrkzpGqcr.FaceImage model.proto < protobuf.bin
        $scope.gen.bin_input = sendPayload;

        //request.setRequestHeader("Content-type", "application/octet-stream;charset=UTF-8");
        request.setRequestHeader("Content-type", "text/plain;charset=UTF-8");
        request.responseType = 'arraybuffer';

        //$(dstImg).addClaas('workingImage').siblings('.spinner').remove().after($("<span class='spinner'>&nbsp;</span>"));

        request.onreadystatechange=function() {
            if (request.readyState==4 && request.status>=200 && request.status<300) {
                var bodyEncodedInString = new Uint8Array(request.response);
                $scope.gen.bin_output = bodyEncodedInString;

                // ---- method for processing from a type ----
                var msgOutput = $scope.gen.proto[methodKeys[0]]['root'].lookupType(
                                    $scope.gen.proto[methodKeys[0]]['methods'][methodKeys[1]]['typeOut']);
                var objRecv = msgOutput.decode(bodyEncodedInString);
                var objRefactor = [];       // what we pass to gen table
                //console.log(msgOutput);
                $.each(msgOutput.fields, function(name,val) {
                    var needCreate = (objRefactor.length == 0);
                    for (var i=0; i<objRecv[name].length; i++) {
                        if (needCreate) {
                            objRefactor.push({});
                        }
                        objRefactor[i][name] = Math.round(1000*objRecv[name][i])/1000;
                        //console.log(inputPayload["frames"][i]);
                        //deref to get prediction target
                        objRefactor[i]["ref"] = $scope.gen.data[inputPayload["frames"][i]["ref"]]["value"];
                    }
                });
                $scope.gen.obj_output = $scope.gen.obj_output.concat(objRefactor);   // append generated objects
                var sliceLen = $scope.gen.obj_output.length-$scope.gen.max_obj_queue;
                if (sliceLen > 0) {
                    $scope.gen.obj_output = $scope.gen.obj_output.slice(sliceLen);
                }

                //genClassTable(objRefactor, dstDiv);
                $scope.gen.transmitting = false;
            }
            else if (request.status >= 400) {
                console.log(request.status);
                clearInterval($scope.gen.trigger_active);
                $scope.gen.trigger_active = 0;
            }
            $scope.$apply();
        }
    	   request.send(sendPayload);
    }

    function vm_gen_tick() {        //always tick, but just call if enabled
        if ($scope.gen.trigger_active) {
            $scope.gen.post_feature(false);
        }
    }

    function vm_init() {
        if ('input_good' in $scope.gen) {
            return; //already been through init
        }
        vm_generate(0.5, 1.0);

        $scope.gen.server = "http://localhost:8882/classify";
        $scope.gen.transmitting = false;
        $scope.gen.proto = {};
        $scope.gen.bin_input = null;
        $scope.gen.bin_output = null;
        $scope.gen.obj_input = [];
        $scope.gen.obj_output = [];
        $scope.gen.max_obj_queue = 10;

        //$("#resultText").hide();

        protobuf_load("assets/model.proto", true);

        $scope.gen.trigger_active = setInterval(vm_gen_tick, 500);

    }
    vm_init();


})
/*
 * Control for selecting different VMs
 */
.controller('VmPicker', function ($scope) {
  $scope.vm = {
    options: [
      '16a0ad6e-4aff-4eb7-ac73-37426abfd16d',
      '803e3807-59b0-4cb8-908e-f77ada7562a3',
      '77049640-f68a-4f61-ad69-07193e4f1446',
      'rcdtxvVvn2b17',
      '340d7db3-8156-4117-8642-c1e006cdecb6',
      '0abc84fd-0d20-47cd-bb90-843db3f2d805',
    ],
    selected: '16a0ad6e-4aff-4eb7-ac73-37426abfd16d'
  };

  $scope.graphs = {
    options: {
        'cpu':{'desc':'CPU Usage', 'path':'charts/cpu_usage'},
        'mem':{'desc':'Memory Usage', 'path':'charts/mem_usage'},
        'net':{'desc':'Net Throughput', 'path':'charts/net_usage'},
    },
    selected: 'cpu'
  };
  $scope.graphSwitch = function(new_graph_sel) {
    $scope.graphs.selected = new_graph_sel;
  }

});


