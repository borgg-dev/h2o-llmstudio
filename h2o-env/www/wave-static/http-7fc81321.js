function c(n){const s=n.regex,e="HTTP/([32]|1\\.[01])",i=/[A-Za-z][A-Za-z0-9-]*/,a={className:"attribute",begin:s.concat("^",i,"(?=\\:\\s)"),starts:{contains:[{className:"punctuation",begin:/: /,relevance:0,starts:{end:"$",relevance:0}}]}},t=[a,{begin:"\\n\\n",starts:{subLanguage:[],endsWithParent:!0}}];return{name:"HTTP",aliases:["https"],illegal:/\S/,contains:[{begin:"^(?="+e+" \\d{3})",end:/$/,contains:[{className:"meta",begin:e},{className:"number",begin:"\\b\\d{3}\\b"}],starts:{end:/\b\B/,illegal:/\S/,contains:t}},{begin:"(?=^[A-Z]+ (.*?) "+e+"$)",end:/$/,contains:[{className:"string",begin:" ",end:" ",excludeBegin:!0,excludeEnd:!0},{className:"meta",begin:e},{className:"keyword",begin:"[A-Z]+"}],starts:{end:/\b\B/,illegal:/\S/,contains:t}},n.inherit(a,{relevance:0})]}}export{c as default};
