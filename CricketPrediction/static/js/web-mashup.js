
function loadLocation(){

   //  var geocode= document.getElementById("location").value;
   //  geocode=encodeURIComponent(geocode);
   //  var data_file = "https://maps.googleapis.com/maps/api/geocode/json?address="+geocode;
    var data_file = "https://cricapi.com/api/matches?apikey=TbY5iS2oz9WFhi01mW2Mk4nWcGv1";
 
    var http_request = new XMLHttpRequest();
    try{
       // Opera 8.0+, Firefox, Chrome, Safari
       http_request = new XMLHttpRequest();
    }catch (e){
       // Internet Explorer Browsers
       try{
          http_request = new ActiveXObject("Msxml2.XMLHTTP");
 
 
       }catch (e) {
 
          try{
             http_request = new ActiveXObject("Microsoft.XMLHTTP");
          }catch (e){
             // Something went wrong
             alert("Your browser broke!");
             return false;
          }
 
       }
    }
 
 
    http_request.open("GET", data_file, true);
 
    http_request.onreadystatechange = function(){
 
    if (http_request.readyState == 4  ){
 
       var jsonObj = JSON.parse(http_request.responseText);    //most most important
 
      //get present date
      var today = new Date();
      var dd = today.getDate()
      var mm = today.getMonth()+1;
      var yyyy = today.getFullYear();
      if(dd<10)
      {
         dd='0'+dd;
      }
      if(mm<10)
      {
         mm='0'+mm;
      }
 
      today = yyyy+'-'+mm+'-'+dd;
 
      var event = new Date(today);
      //console.log(event.toString());
      // expected output: Wed Oct 05 2011 16:48:00 GMT+0200 (CEST)
      // (note: your timezone may vary)
 
      //console.log(event.toISOString());
      var present_date = event.toISOString();
      // expected output: 2011-10-05T14:48:00.000Z
 
 
 
 
 
 
 
 // $.parseJSON will parse the txt (JSON) and convert it to an
 // JavaScript object. After its call, it gets the employees property
 // and sets it to the employees variable
 //var matches = $.parseJSON( txt ).matches;
 
 var $table = $( "<table></table>" );
 var $line = $( "<tr></tr>" );
 //$line.append( $("<td></td>").html("Upcoming Matches"));
 //$table.append( $line );
 $line.append( $("<td></td>").html("Date"));
 $line.append( $("<td></td>").html("Team 1"));
 $line.append( $("<td></td>").html("     Vs      "));
 $line.append( $("<td></td>").html(" Team2  "));
 $line.append( $("<td></td>").html("Select"));
 $table.append( $line );
 for (var i = 0; i < jsonObj.matches.length; i++ ) {
  if((jsonObj.matches[i].type == 'ODI')  && (jsonObj.matches[i].date>=present_date)){
 
    //date date1=jsonObj.matches[i].date;
    var mat = jsonObj.matches[i];
 
 
 
    $line = $( "<tr></tr>" );
 
    //var r= $('<input type="button" value="new button"/>');
    var button = document.createElement('button');
    button.setAttribute('type', 'button');
    // button.className('abc');
    button.innerHTML = "Select";
    button.id=mat['team-1']+"#"+mat['team-2'];
 ///mere changes 25 june
   //console.log(button.id)
   //console.log(this.button.id)
 //   button.addEventListener ("click", function() {
 //   displayPlayer(button.id);
 // });
     button.onclick = function() {displayPlayer(this.id)};
 
    $line.append( $("<td></td>").html(jsonObj.matches[i].date));
    $line.append( $( "<td></td>" ).html( mat['team-1'] ) );
    $line.append( $("<td></td>").html("     Vs      "));
    $line.append( $( "<td></td>" ).html( mat['team-2'] ) );
    // $line.append( $( "<td></td>" ).html(document.body.appendChild(button)) );
    // $(function(){
     // $('button').on('click',function(){
 
 //         $("hey").append(r);
 //     });
 // });
 
 
 
    $line.append( $( "<td></td>" ).html(button));
    // button.onclick = function() {
    //   console.log((mat['team-1']));
    //   displayPlayer(mat['team-1'],mat['team-2']);
    // };
 
 
 
    $table.append( $line );
 }
 }
 
 
 $table.appendTo( document.body );
 
 // if you want to insert this table in a div with id attribute
 // set as "myDiv", you can do this:
 $table.appendTo( $( "#myDiv" ) );
 
    }
 }
    http_request.send();
 }
 
 
 
 
 // /my work to diplay countdrmolvdn
 // var a,b;
 function displayPlayer(button_id){
   //console.log(button_id);
   //isme split ho gyaa
   var team = button_id.split('#');  //declared globalyy
   console.log(team[0] + "  " + team[1]);
   localStorage.setItem("a", team[0]);
   localStorage.setItem("b", team[1]);

 
   // document.getElementById('India').style.display = "unset";
   // // document.getElementById('Pakistan').style.display = "unset";
   // a = team[0];
   // b = team[1];
   // console.log(a)
   //document.getElementById("India").style.visibility='visible';
   //document.getElementById("Pakistan").style.visibility='visible';


   // window.location.href = "table.html";
   window.location.href = 'http://localhost:8000/displayPlayer/';
   // console.log(a);
   console.log("jweodnwcikwncin");
 
  }
 
 
 function showTable(){
   var a = localStorage.getItem('a');
   var b = localStorage.getItem('b');
  console.log(a);
   document.getElementById(a).style.display='unset';
   document.getElementById(b).style.display='unset';
 }
 