************************
OpenDataVal Leaderboards
************************

Save and upload the results of your DataEvaluator `here <https://docs.google.com/forms/d/e/1FAIpQLSfDzkI-gRKRCvNEmY-VdRh2mZJ5ls8w1baLd-autGbQ7A61bA/viewform?usp=sf_link>`_.

::

    from opendataval.experiment.save_dataval
    save_dataval(dataevaluator, fetcher, "output.csv")




.. raw:: html


    <script src="_static/leaderboards.js"></script>
    <link rel="stylesheet" href="_static/leaderboards.css">
    <script src="_static/leaderboards.js"></script>

    <button class="collapsible"> PlaceHolder </button>
    <div class="collapsible-content">
        <h4> Place Holder 1 </h4>
        <table id="excelDataTable" class="table-wrapper docutils container">
            <tr>
                <th id="col1" class='sortable'>
                    <span class="custom-tooltip">
                        col1
                        <span class="tooltiptext"> Lorem ipsum filler yada yada </span>
                    </span>
                </th>

                <th id="col2" class='sortable'>col2</th>
                <th id="col3" class='sortable'>col3</th>
            </tr>
        </table>

        <h4> Place Holder 1 </h4>
        <table id="excelDataTable3" class="table-wrapper docutils container">
            <tr>
                <th id="col1" class='sortable'>
                    <span class="custom-tooltip">
                        col1
                        <span class="tooltiptext"> Lorem ipsum filler yada yada </span>
                    </span>
                </th>

                <th id="col2" class='sortable'>col2</th>
                <th id="col3" class='sortable'>col3</th>
            </tr>
        </table>
    </div>

    <button class="collapsible"> Iris Challenge </button>
    <div class="collapsible-content">
        <h4> Noisy Indices Detection F1 Score </h4>
        <table id="iristable" class="table-wrapper docutils container">
            <tr>
                <th id="dataval_name" class='sortable'>
                <span class="custom-tooltip">
                    Data Evaluator Name
                    <span class="tooltiptext"> Name of the Evaluator </span>
                </span>
                </th>

                <th id="KMeans F1" class='sortable'>
                <span class="custom-tooltip">
                    Noisy F1 Score
                    <span class="tooltiptext"> F1 score of the data evaluator in identifying noisy data with a 2Means classifier </span>
                </span>
                </th>
            </tr>
        </table>
    </div>

    <script type="text/javascript">
        window.onload =  function(){
            fetchLeaderboards('table1', 'excelDataTable');
            fetchLeaderboards('table1', 'excelDataTable3');
            fetchLeaderboards('iristable', 'iristable');
        };
    </script>
    <script src="https://kit.fontawesome.com/ecc02ef754.js" crossorigin="anonymous"></script>
    <script src="_static/leaderboards.js"></script>

