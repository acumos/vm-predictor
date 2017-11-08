/**
 vmpredictor.js - angular backend for interaction with vm predictor demo

 E. Zavesky 8/8/17 adapted for Acumos
 */
 
"use strict";

/*
 * Control for selecting different date range
 */

angular.module('vmpredictor', ['ui.bootstrap']).controller('DatepickerPopupCtrl', function ($scope) {
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


/**
 * convert base64/URLEncoded data component to raw binary data held in a string
 *
 * Stoive, http://stackoverflow.com/questions/4998908/convert-data-uri-to-file-then-append-to-formdata
 */
function dataURItoBlob(dataURI) {
    // convert base64/URLEncoded data component to raw binary data held in a string
    var byteString;
    if (dataURI.split(',')[0].indexOf('base64') >= 0)
        byteString = atob(dataURI.split(',')[1]);
    else
        byteString = unescape(dataURI.split(',')[1]);

    // separate out the mime component
    var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

    // write the bytes of the string to a typed array
    var ia = new Uint8Array(byteString.length);
    for (var i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ia], {type:mimeString});
}

