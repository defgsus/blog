
Just a small notebook extension to pre-load a local copy of 
[jQuery.DataTables](https://datatables.net) on start of a notebook.

The *copy* contains the un-minified CSS and the required images.
And an adjusted `datatables-dark.css` which is put into the html header
by the `main.js` file (at the bottom).

To install, move to parent directory and say:
```shell script
jupyter nbextension install --user datatables-offline
```