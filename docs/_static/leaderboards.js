
BASE = "https://35.208.129.173"

async function fetchLeaderboards(table, target_id){ // path
    console.log(table);
    uri = new URL(`/leaderboards/${table}`, BASE)
    data = await fetch(uri, {method: "GET"}).then(response => response.json());

    table = document.getElementById(target_id);
    buildHtmlTable(data.table, table);
}


function sortTable(col, table_name) {
    var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
    table = document.getElementById(table_name);
    switching = true;
    // Set the sorting direction to ascending:
    dir = "asc";
    /* Make a loop that will continue until
    no switching has been done: */
    while (switching) {
      // Start by saying: no switching is done:
      switching = false;
      rows = table.rows;
      /* Loop through all table rows (except the
      first, which contains table headers): */
      for (i = 1; i < (rows.length - 1); i++) {
        // Start by saying there should be no switching:
        shouldSwitch = false;
        /* Get the two elements you want to compare,
        one from current row and one from the next: */
        x = rows[i].getElementsByTagName("TD")[col];
        y = rows[i + 1].getElementsByTagName("TD")[col];
        /* Check if the two rows should switch place,
        based on the direction, asc or desc: */
        if (dir == "asc") {
          if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
            // If so, mark as a switch and break the loop:
            shouldSwitch = true;
            break;
          }
        } else if (dir == "desc") {
          if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
            // If so, mark as a switch and break the loop:
            shouldSwitch = true;
            break;
          }
        }
      }
      if (shouldSwitch) {
        /* If a switch has been marked, make the switch
        and mark that a switch has been done: */
        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
        switching = true;
        // Each time a switch is done, increase this count by 1:
        switchcount ++;
      } else {
        /* If no switching has been done AND the direction is "asc",
        set the direction to "desc" and run the while loop again. */
        if (switchcount == 0 && dir == "asc") {
          dir = "desc";
          switching = true;
        }
      }
    }

    const ARROW_UP = ' <i class="fas fa-arrow-up" aria-hidden="true"></i>';
    const ARROW_DOWN = ' <i class="fas fa-arrow-down" aria-hidden="true"></i>';

    Array.from(table.rows[0].cells).forEach((x) => {
        x.innerHTML = x.innerHTML.replace(ARROW_UP, '').replace(ARROW_DOWN, '');
    })
    table.rows[0].cells[col].innerHTML += (dir == "asc") ? ARROW_DOWN : ARROW_UP;
}



var _tr_ = document.createElement('tr'),
_th_ = document.createElement('th'),
_td_ = document.createElement('td');

// Builds the HTML Table out of data json data from Ivy restful service.
function buildHtmlTable(data, table) {
    let columns = Array.from(table.rows[0].cells).map((x, i) => {
        if (x.classList.contains('sortable')){
            x.onclick = function() { sortTable(i, table.id) };
        }
        return x.id;
    });


    data.forEach((row) => {
        var tr = _tr_.cloneNode(false);

        columns.forEach((col) => {
            var td = _td_.cloneNode(false);
            var cellValue = row[col];
            td.appendChild(document.createTextNode(row[col] || ''));
            tr.appendChild(td);
        })

        table.appendChild(tr);
    })
}


var coll = document.getElementsByClassName("collapsible");
for (var i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.maxHeight){
      content.style.maxHeight = null;
    } else {
      content.style.maxHeight = content.scrollHeight + "px";
    }
  });
}