//show separator images

function showSep1(){
    document.getElementById("sep1").style.display="block";
    document.getElementById("separator-2").style.display="none";
    document.getElementById("separator-3").style.display="none";
    console.log("sep1 checked")

    document.getElementById("gas-out1").style.display="none";
    document.getElementById("gas-out2").style.display="none";
    document.getElementById("gas-out3").style.display="none";
    document.getElementById("oil-out1").style.display="none";
    document.getElementById("oil-out2").style.display="none";
    document.getElementById("oil-out3").style.display="none";
    document.getElementById("summary").style.display="none";
}

function showSep2() {
    document.getElementById("sep1").style.display="block";
    document.getElementById("separator-2").style.display="block";
    document.getElementById("separator-3").style.display="none";
    console.log("sep2 checked")

    document.getElementById("gas-out1").style.display="none";
    document.getElementById("gas-out2").style.display="none";
    document.getElementById("gas-out3").style.display="none";
    document.getElementById("oil-out1").style.display="none";
    document.getElementById("oil-out2").style.display="none";
    document.getElementById("oil-out3").style.display="none";
    document.getElementById("summary").style.display="none";
}



function showSep3(){
    document.getElementById("sep1").style.display="block";
    document.getElementById("separator-3").style.display="block";
    document.getElementById("separator-2").style.display="block";
    console.log("sep3 checked")

    document.getElementById("gas-out1").style.display="none";
    document.getElementById("gas-out2").style.display="none";
    document.getElementById("gas-out3").style.display="none";
    document.getElementById("oil-out1").style.display="none";
    document.getElementById("oil-out2").style.display="none";
    document.getElementById("oil-out3").style.display="none";
    document.getElementById("summary").style.display="none";
}


// show input tables + blur background

function showInput1() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table1-input").style.display="block";
    console.log("clicked separator 1 image");
}

function showInput2() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table2-input").style.display="flex";
    console.log("clicked separator 2 image");
}

function showInput3() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table3-input").style.display="flex";
    console.log("clicked separator 3 image");
}

// exit blurred background

function exitBlur2() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table2-input").style.display="none";
    console.log("clicked sep2 exit blur");
}

function exitBlur1() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table1-input").style.display="none";
    console.log("clicked sep1 exit blur");
}

function exitBlur3() {
    var blur1 = document.getElementById("main");
    var blur2 = document.getElementById("title-head")
    var blur3 = document.getElementById("initial-input-div")
    var blur4 = document.getElementById("note1")
    blur1.classList.toggle('active')
    blur2.classList.toggle('active')
    blur3.classList.toggle('active')
    blur4.classList.toggle('active')
    document.getElementById("table3-input").style.display="none";
    console.log("clicked sep3 exit blur");
}


// function when hitting calculate

function showResults() {
    if(document.getElementById('stage1').checked) {
        console.log("sep1 selected")
        document.getElementById("gas-out1").style.display="flex";
        document.getElementById("gas-out2").style.display="none";
        document.getElementById("gas-out3").style.display="none";

        document.getElementById("oil-out1").style.display="flex";
        document.getElementById("oil-out2").style.display="none";
        document.getElementById("oil-out3").style.display="none";

    }else if(document.getElementById('stage2').checked) {
        console.log("sep2 selected")
        document.getElementById("gas-out1").style.display="flex";
        document.getElementById("gas-out2").style.display="flex";
        document.getElementById("gas-out3").style.display="none";

        document.getElementById("oil-out1").style.display="flex";
        document.getElementById("oil-out2").style.display="flex";
        document.getElementById("oil-out3").style.display="none";

    } else if(document.getElementById('stage3').checked) {
        console.log("sep3 selected")
        document.getElementById("gas-out1").style.display="flex";
        document.getElementById("gas-out2").style.display="flex";
        document.getElementById("gas-out3").style.display="flex";

        document.getElementById("oil-out1").style.display="flex";
        document.getElementById("oil-out2").style.display="flex";
        document.getElementById("oil-out3").style.display="flex";
    }

    document.getElementById("summary").style.display="block";

}



//maybe need to add promises for return

// taking form data and preventing submit/reload issue using jQuery/ajax


$(document).ready(function() {

    $('#input-form').submit(function(event) {


        if ($("input[name=stage]:checked").val() == "1") {
        $.ajax({
            data: {
                stage : $('.stage:checked').val(),
                SG: $('#SG').val(), 
                C7_plus_SG: $('#C7_plus_SG').val(),
                C7_plus_MW: $('#C7_plus_MW').val(),
                C1: $('#C1').val(),
                C2: $('#C2').val(),
                C3: $('#C3').val(),
                iC4: $('#iC4').val(),
                C4: $('#C4').val(),
                iC5: $('#iC5').val(),
                C5: $('#C5').val(),
                C6: $('#C6').val(),
                C7: $('#C7').val(),
                CO2: $('#CO2').val(),
                N2: $('#N2').val(),
                units: $('.units:checked').val(),
                P: $('#P').val(),
                T: $('#T').val()
            },
            type: 'POST',
            async: false,
            url: "/EOS/",
            success: function(data){
                console.log("posted");
                $("#main").html(data);

                $("#main").css("display", "flex");
                $("#gas-out1").css("display", "flex");
                $("#oil-out1").css("display", "flex");
                $("#sep1").css("display", "block");
                $("#summary").css("display", "block");
            }
        });
        };


        if ($("input[name=stage]:checked").val() == "2") {
            $.ajax({
                data: {
                    stage : $('.stage:checked').val(),
                    SG: $('#SG').val(), 
                    C7_plus_SG: $('#C7_plus_SG').val(),
                    C7_plus_MW: $('#C7_plus_MW').val(),
                    C1: $('#C1').val(),
                    C2: $('#C2').val(),
                    C3: $('#C3').val(),
                    iC4: $('#iC4').val(),
                    C4: $('#C4').val(),
                    iC5: $('#iC5').val(),
                    C5: $('#C5').val(),
                    C6: $('#C6').val(),
                    C7: $('#C7').val(),
                    CO2: $('#CO2').val(),
                    N2: $('#N2').val(),
                    units: $('.units:checked').val(),
                    P: $('#P').val(),
                    T: $('#T').val(),
                    P2: $('#P2').val(),
                    T2: $('#T2').val()

                },
                type: 'POST',
                async: false,
                url: "/EOS/",
                success: function(data){
                    console.log("posted");
                    $("#main").html(data);
    
                    $("#main").css("display", "flex");
                    $("#gas-out1").css("display", "flex");
                    $("#oil-out1").css("display", "flex");
                    $("#sep1").css("display", "block");
                    $("#summary").css("display", "block");

                    $("#gas-out2").css("display", "flex");
                    $("#oil-out2").css("display", "flex");
                    $("#separator-2").css("display", "flex");
                }
            });
            };

            if ($("input[name=stage]:checked").val() == "3") {
                $.ajax({
                    data: {
                        stage : $('.stage:checked').val(),
                        SG: $('#SG').val(), 
                        C7_plus_SG: $('#C7_plus_SG').val(),
                        C7_plus_MW: $('#C7_plus_MW').val(),
                        C1: $('#C1').val(),
                        C2: $('#C2').val(),
                        C3: $('#C3').val(),
                        iC4: $('#iC4').val(),
                        C4: $('#C4').val(),
                        iC5: $('#iC5').val(),
                        C5: $('#C5').val(),
                        C6: $('#C6').val(),
                        C7: $('#C7').val(),
                        CO2: $('#CO2').val(),
                        N2: $('#N2').val(),
                        units: $('.units:checked').val(),
                        P: $('#P').val(),
                        T: $('#T').val(),
                        P2: $('#P2').val(),
                        T2: $('#T2').val(),
                        T3: $('#T3').val(),
                        P3: $('#P3').val()
                    },
                    type: 'POST',
                    async: false,
                    url: "/EOS/",
                    success: function(data){
                        console.log("posted");
                        $("#main").html(data);
        
                        $("#main").css("display", "flex");
                        $("#gas-out1").css("display", "flex");
                        $("#oil-out1").css("display", "flex");
                        $("#sep1").css("display", "block");
                        $("#summary").css("display", "block");
    
                        $("#gas-out2").css("display", "flex");
                        $("#oil-out2").css("display", "flex");
                        $("#separator-2").css("display", "flex");

                        $("#gas-out3").css("display", "flex");
                        $("#oil-out3").css("display", "flex");
                        $("#separator-3").css("display", "flex");
                    }
                });
                };

        

        $("#main").css("display", "flex");
        $("#gas-out1").css("display", "flex");
        $("#oil-out1").css("display", "flex");
        $("#sep1").css("display", "block");
        $("#summary").css("display", "block");
        console.log($('.stage:checked').val());
        console.log($('.units:checked').val());
        //preventing the submit
        event.preventDefault();
    });

});