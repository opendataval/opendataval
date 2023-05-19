************************
OpenDataVal Leaderboards
************************


.. raw:: html

    <link rel="stylesheet" href="_static/leaderboards.css">
    <script src="_static/leaderboards.js"></script>

    <button class="collapsible"> Title </button>
    <div class="collapsible-content">
        <table id="excelDataTable" class="table-wrapper docutils container">
            <tr>
                <th id="col1" class='sortable'>
                    <span class="custom-tooltip">
                        col1
                        <span class="tooltiptext"> "Lorem ipsum filler yada yada" </span>
                    </span>
                </th>

                <th id="col2" class='sortable'>col2</th>
                <th id="col3" class='sortable'>col3</th>
            </tr>
        </table>

        <table id="excelDataTable3" class="table-wrapper docutils container">
            <tr>
                <th id="col1" class='sortable'>
                    <span class="custom-tooltip">
                        col1
                        <span class="tooltiptext"> "Lorem ipsum filler yada yada" </span>
                    </span>
                </th>

                <th id="col2" class='sortable'>col2</th>
                <th id="col3" class='sortable'>col3</th>
            </tr>
        </table>

    </div>

    <button class="collapsible"> Title2 </button>
    <div class="collapsible-content">
        <table id="excelDataTable2" class="table-wrapper docutils container">
            <tr>
                <th id="col1" class='sortable'>
                    <span class="custom-tooltip">
                        col1
                        <span class="tooltiptext"> "Lorem ipsum filler yada yada" </span>
                    </span>
                </th>

                <th id="col2" class='sortable'>col2</th>
                <th id="col3" class='sortable'>col3</th>
            </tr>
        </table>
    </div>

    <script type="text/javascript">
        window.onload =  function(){
            fetchLeaderboards('Table1', 'excelDataTable');
            fetchLeaderboards('Table1', 'excelDataTable2');
            fetchLeaderboards('Table1', 'excelDataTable3');
        };
    </script>
    <script src="https://kit.fontawesome.com/ecc02ef754.js" crossorigin="anonymous"></script>
    <script src="_static/leaderboards.js"></script>

