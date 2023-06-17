# opendataval docs
> folder housing opendataval documentation built by sphinx.

## Extending

In the `leaderboards.rst`, we can write raw HTML. To add a new challenge table, first append to the file a collapsible button and div for the content. Now inside that div  we have to define a table as follows. Note that `tableid` is important and will be used later on:
```html
<button class="collapsible"> Button Name </button>
<div class="collapsible-content">
    <h4> Table Name </h4>
    <table id="tableid" class="table-wrapper docutils container">
        ...
    </table>
</div>
```

Now we add the content of our table. Because of the api, we only need to add the headers of the table. Inside the `table` (where the ellipsis is) add the following. Note that the first row should ALWAYS be the first entry as this is the content of name, description and hyperlink of the data evaluator:
```html
<th id="dataeval" class='sortable'>
    <span class="custom-tooltip">
        Data Evaluator Name
        <span class="tooltiptext"> Name of the Evaluator </span>
    </span>
</th>
```

Next we add the additional columns. These columns should match the output of the backend. In other words the return object of the backend is a `list[dict[str, Any]]` and table header `id` should match a key of the `dict[str, Any]` exactly. All `dict`s of the list should have the same keys. Append the following directly below the `Data Evaluator Name` entry.

```html
<th id="Key Name 1" class='sortable'>
<span class="custom-tooltip">
    Column name 1
    <span class="tooltiptext"> Optional description </span>
</span>
</th>

<th id="Key Name 2">
    Column name 2
</th>
```

Finally at the bottom of the file there is a `<script type="text/javascript">` that will query the database on load. Replace `'ChallengeName'` with the name of your challenge in the backend. Modify as the following:
```html
<script type="text/javascript">
    window.onload =  function(){
        fetchLeaderboards('ChallengeIris', 'iristable');
        fetchLeaderboards('ChallengeName', 'tableid'); // added content
    };
</script>
```

Congrats, you just added a new challenge to the frontend. Don't forget to rebuild the site by pushing to the website with the GitHub action.