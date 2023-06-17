************************
OpenDataVal Leaderboards
************************

Save and upload the data values for a Challenge task `here <https://docs.google.com/forms/d/e/1FAIpQLSfDzkI-gRKRCvNEmY-VdRh2mZJ5ls8w1baLd-autGbQ7A61bA/viewform?usp=sf_link>`_. After running your experiment and computing the data values, run the following function to receive a csv file.
Upload the csv file to the above Google Form to appear to compare your DataEvaluator and to appear on the leaderboards.
All datasets that begin with the prefix ``'challenge-*'`` are valid challenges that can be submitted.

::

    from opendataval.experiment import save_dataval
    from opendataval.dataloader import DataFetcher
    fetcher = DataFetcher("challenge-*")
    save_dataval(dataevaluator, fetcher, "output.csv")




.. raw:: html


    <script src="_static/leaderboards.js"></script>
    <link rel="stylesheet" href="_static/leaderboards.css">
    <script src="_static/leaderboards.js"></script>


    <button class="collapsible"> Iris Challenge </button>
    <div class="collapsible-content">
        <h4> Noisy Indices Detection F1 Score </h4>
        <table id="iristable" class="table-wrapper docutils container">
            <tr>
                <th id="dataeval" class='sortable'>
                    <span class="custom-tooltip">
                        Data Evaluator Name
                        <span class="tooltiptext"> Name of the Evaluator </span>
                    </span>
                </th>

                <th id="F1 Kmeans" class='sortable'>
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
            fetchLeaderboards('ChallengeIris', 'iristable');
        };
    </script>
    <script src="https://kit.fontawesome.com/ecc02ef754.js" crossorigin="anonymous"></script>
    <script src="_static/leaderboards.js"></script>

