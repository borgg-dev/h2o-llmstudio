import{_ as u,r as a,j as L}from"./index-34f83032.js";import{u as T}from"./useThemeProps-99155e78.js";const h=e=>({components:{MuiLocalizationProvider:{defaultProps:{localeText:u({},e)}}}}),p={previousMonth:"Previous month",nextMonth:"Next month",openPreviousView:"open previous view",openNextView:"open next view",calendarViewSwitchingButtonAriaLabel:e=>e==="year"?"year view is open, switch to calendar view":"calendar view is open, switch to year view",start:"Start",end:"End",cancelButtonLabel:"Cancel",clearButtonLabel:"Clear",okButtonLabel:"OK",todayButtonLabel:"Today",clockLabelText:(e,t,n)=>"Select ".concat(e,". ").concat(t===null?"No time selected":"Selected time is ".concat(n.format(t,"fullTime"))),hoursClockNumberText:e=>"".concat(e," hours"),minutesClockNumberText:e=>"".concat(e," minutes"),secondsClockNumberText:e=>"".concat(e," seconds"),openDatePickerDialogue:(e,t)=>e&&t.isValid(t.date(e))?"Choose date, selected date is ".concat(t.format(t.date(e),"fullDate")):"Choose date",openTimePickerDialogue:(e,t)=>e&&t.isValid(t.date(e))?"Choose time, selected time is ".concat(t.format(t.date(e),"fullTime")):"Choose time",timeTableLabel:"pick time",dateTableLabel:"pick date"},v=p;h(p);const b=a.createContext(null);function k(e){const t=T({props:e,name:"MuiLocalizationProvider"}),{children:n,dateAdapter:s,dateFormats:c,dateLibInstance:l,locale:d,adapterLocale:i,localeText:r}=t,o=a.useMemo(()=>new s({locale:i!=null?i:d,formats:c,instance:l}),[s,d,i,c,l]),m=a.useMemo(()=>({minDate:o.date("1900-01-01T00:00:00.000"),maxDate:o.date("2099-12-31T00:00:00.000")}),[o]),x=a.useMemo(()=>({utils:o,defaultDates:m,localeText:u({},v,r!=null?r:{})}),[m,o,r]);return L(b.Provider,{value:x,children:n})}export{k as L,b as M};
