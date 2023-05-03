
const API_URL = "http://localhost"
const base = new URL(`${API_URL}/leaderboards`);

async function fetchLeaderboards(uri, target_id){ // path
    uri = new URL(uri, base)
    data = await fetch(uri, {method: "GET"}).then(response => response.json());

    table = document.getElementById(target_id);
    buildHtmlTable(data.table, table);
}


var _tr_ = document.createElement('tr'),
_th_ = document.createElement('th'),
_td_ = document.createElement('td');

// Builds the HTML Table out of data json data from Ivy restful service.
function buildHtmlTable(data, table) {
    let columns = Array.from(table.rows[0].cells).map(x => x.id);
    console.log(columns);
    data.forEach((row) => {
        var tr = _tr_.cloneNode(false);

        columns.forEach((col) => {
            var td = _td_.cloneNode(false);
            var cellValue = row[col];
            console.log(cellValue);
            td.appendChild(document.createTextNode(row[col] || ''));
            tr.appendChild(td);
        })

        table.appendChild(tr);
    })
}

